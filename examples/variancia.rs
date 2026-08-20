use adaptive_parallel_insertion_merge as api;
use api::{multimerge, PimConfig};
use std::time::Instant;

const AMOSTRAS: usize = 41;
const DIC: usize = 1_048_576;

fn dicionario() -> &'static [String] {
    let v: Vec<String> = (0..DIC).map(|i| format!("PIM-CHAVE-{i:08X}")).collect();
    Box::leak(v.into_boxed_slice())
}

fn inverso_rep(n: usize, rep: usize, dic: &'static [String]) -> Vec<&'static str> {
    let mut v = Vec::with_capacity(n);
    let mut i = dic.len();
    while v.len() < n && i > 0 {
        i -= 1;
        for _ in 0..rep {
            if v.len() < n {
                v.push(dic[i].as_str());
            }
        }
    }
    while v.len() < n {
        v.push(dic[0].as_str());
    }
    v
}

fn coleta<T: Ord + Copy + Send + Sync>(
    base: &[T],
    v: &mut Vec<T>,
    c: Option<PimConfig>,
) -> Vec<f64> {
    let mut t = Vec::with_capacity(AMOSTRAS);
    for _ in 0..AMOSTRAS {
        v.copy_from_slice(base);
        let ini = Instant::now();
        match c {
            None => multimerge::multi_merge_sort(v),
            Some(k) => api::pim_sort_with_config(v, k),
        }
        let ms = ini.elapsed().as_secs_f64() * 1e3;
        assert!(v.windows(2).all(|w| w[0] <= w[1]), "saida nao ordenada");
        t.push(ms);
    }
    t
}

fn quantil(ord: &[f64], q: f64) -> f64 {
    let i = ((ord.len() - 1) as f64 * q).round() as usize;
    ord[i]
}

/// Maior salto relativo entre amostras consecutivas ordenadas.
///
/// Numa distribuicao continua os saltos sao pequenos e parecidos. Se houver
/// DOIS REGIMES -- execucoes rapidas e execucoes lentas -- aparece um degrau
/// isolado, e a posicao dele diz quantas amostras cairam em cada regime.
fn maior_degrau(ord: &[f64]) -> (f64, usize) {
    let mut melhor = (0.0f64, 0usize);
    for i in 1..ord.len() {
        let d = 100.0 * (ord[i] - ord[i - 1]) / ord[i - 1];
        if d > melhor.0 {
            melhor = (d, i);
        }
    }
    melhor
}

fn histograma(ord: &[f64]) {
    let (lo, hi) = (ord[0], ord[ord.len() - 1]);
    if hi <= lo {
        return;
    }
    const FAIXAS: usize = 12;
    let mut conta = [0usize; FAIXAS];
    for &x in ord {
        let i = (((x - lo) / (hi - lo)) * (FAIXAS - 1) as f64).round() as usize;
        conta[i.min(FAIXAS - 1)] += 1;
    }
    let pico = *conta.iter().max().unwrap_or(&1);
    for (i, &c) in conta.iter().enumerate() {
        let ini = lo + (hi - lo) * i as f64 / FAIXAS as f64;
        let barras = if pico == 0 { 0 } else { c * 40 / pico };
        println!(
            "    {:>8.1}ms |{:<40}| {}",
            ini,
            "#".repeat(barras),
            if c > 0 { c.to_string() } else { String::new() }
        );
    }
}

fn analisa<T: Ord + Copy + Send + Sync>(rotulo: &str, base: &[T], v: &mut Vec<T>, c: Option<PimConfig>) {
    let t = coleta(base, v, c);
    let mut ord = t.clone();
    ord.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let med = quantil(&ord, 0.5);
    let (degrau, pos) = maior_degrau(&ord);

    println!("\n  --- {rotulo} ---");
    println!(
        "    min={:.1}  p25={:.1}  mediana={:.1}  p75={:.1}  max={:.1}  (max/min = {:.2}x)",
        ord[0],
        quantil(&ord, 0.25),
        med,
        quantil(&ord, 0.75),
        ord[ord.len() - 1],
        ord[ord.len() - 1] / ord[0]
    );
    println!(
        "    maior degrau entre amostras vizinhas: {:.1}% na posicao {}/{}",
        degrau,
        pos,
        ord.len()
    );
    if degrau > 20.0 {
        println!(
            "    BIMODAL: {} amostras rapidas (ate {:.1}ms) e {} lentas (a partir de {:.1}ms)",
            pos,
            ord[pos - 1],
            ord.len() - pos,
            ord[pos]
        );
    } else {
        println!("    espalhamento CONTINUO: nenhum degrau isolado");
    }
    histograma(&ord);

    // Ha correlacao com a ORDEM de execucao? Se as primeiras forem lentas e as
    // ultimas rapidas, e aquecimento, nao propriedade do algoritmo.
    let metade = t.len() / 2;
    let ini: f64 = t[..metade].iter().sum::<f64>() / metade as f64;
    let fim: f64 = t[metade..].iter().sum::<f64>() / (t.len() - metade) as f64;
    println!(
        "    primeira metade={:.1}ms  segunda metade={:.1}ms  ({:+.1}%)",
        ini,
        fim,
        100.0 * (fim - ini) / ini
    );
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);
    let dic = dicionario();

    println!("VARIANCIA — {} amostras por configuracao | {} threads",
             AMOSTRAS, rayon::current_num_threads());
    println!("O caso rep=9 mostrou dispersao de 70-80% no PIM contra 10% no multimerge.");
    println!("Dispersao nao se corrige com parametro: a pergunta e se ha DOIS REGIMES");
    println!("de execucao ou um espalhamento continuo.\n");
    println!("Leitura:");
    println!("  degrau > 20% em posicao intermediaria -> bimodal, algo discreto alterna");
    println!("  espalhamento continuo                 -> contencao ou ruido externo");
    println!("  segunda metade muito menor            -> aquecimento, nao e o algoritmo");

    for rep in [9usize, 30] {
        let base = inverso_rep(n, rep, dic);
        let runs = api::detect_global_trend(&base).len();
        let mut v: Vec<&'static str> = base.to_vec();

        println!("\n{}", "=".repeat(70));
        println!("&str INVERSO rep={rep}   runs={}  run_med={}",
                 runs, base.len() / runs.max(1));
        println!("{}", "=".repeat(70));

        analisa("multimerge (referencia estavel)", &base, &mut v, None);
        analisa("pim default", &base, &mut v, Some(PimConfig::default()));

        let mut so_runs = PimConfig::default();
        so_runs.run_medio_minimo = 2;
        analisa("pim forcando rota de RUNS", &base, &mut v, Some(so_runs));

        let mut so_blocos = PimConfig::default();
        so_blocos.run_medio_minimo = 1_000_000;
        analisa("pim forcando rota de BLOCOS", &base, &mut v, Some(so_blocos));
    }

    println!("\n{}", "=".repeat(70));
    println!("Se so UMA das duas rotas for bimodal, a variancia esta nela e o proximo");
    println!("passo e instrumentar aquele caminho. Se as duas forem, a causa e comum.");
    println!("Se o multimerge tambem for bimodal, o problema e a maquina, nao o codigo.");
}
