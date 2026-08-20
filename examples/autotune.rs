use adaptive_parallel_insertion_merge as api;
use api::{multimerge, PimConfig};
use std::time::Instant;

const REPS: usize = 7;
const DIC: usize = 1_048_576;
const SEM_GALOPE: usize = usize::MAX / 4;

// ============================================================
// VARIAVEIS DA BUSCA
// ============================================================

struct Var {
    nome: &'static str,
    valores: &'static [usize],
    aplica: fn(&mut PimConfig, usize),
    mostra: fn(usize) -> String,
}

fn v_simples(x: usize) -> String { x.to_string() }
fn v_galope(x: usize) -> String {
    if x >= SEM_GALOPE { "off".into() } else { x.to_string() }
}
fn v_bool(x: usize) -> String { if x == 0 { "false".into() } else { "true".into() } }
fn v_auto(x: usize) -> String { if x == 0 { "auto".into() } else { x.to_string() } }

const VARS: &[Var] = &[
    Var {
        nome: "run_medio_minimo",
        valores: &[0, 2, 4, 8, 16, 32, 64, 128],
        aplica: |c, x| c.run_medio_minimo = x,
        mostra: v_simples,
    },
    Var {
        nome: "folha_override",
        valores: &[0, 512, 1024, 2048, 4096, 8192],
        aplica: |c, x| c.folha_override = x,
        mostra: v_auto,
    },
    Var {
        nome: "min_gallop",
        valores: &[7, 16, 24, 64, 256, SEM_GALOPE],
        aplica: |c, x| c.min_gallop = x,
        mostra: v_galope,
    },
    Var {
        nome: "runs_via_kway",
        valores: &[0, 1],
        aplica: |c, x| c.runs_via_kway = x != 0,
        mostra: v_bool,
    },
    Var {
        nome: "run_curto",
        valores: &[2, 8, 32, 128, 512],
        aplica: |c, x| c.run_curto = x,
        mostra: v_simples,
    },
    Var {
        nome: "min_segmento_pway",
        valores: &[1024, 2048, 4096, 16384],
        aplica: |c, x| c.min_segmento_pway = x,
        mostra: v_simples,
    },
];

// ============================================================
// MEDICAO
// ============================================================

fn mediana(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn mede<T: Ord + Copy + Send + Sync>(base: &[T], v: &mut Vec<T>, c: Option<PimConfig>) -> f64 {
    let mut t = Vec::with_capacity(REPS);
    for _ in 0..REPS {
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
    mediana(&mut t)
}

fn descreve(c: &PimConfig) -> String {
    VARS.iter()
        .map(|v| {
            let atual = match v.nome {
                "run_medio_minimo" => c.run_medio_minimo,
                "folha_override" => c.folha_override,
                "min_gallop" => c.min_gallop,
                "runs_via_kway" => usize::from(c.runs_via_kway),
                "run_curto" => c.run_curto,
                _ => c.min_segmento_pway,
            };
            format!("{}={}", v.nome, (v.mostra)(atual))
        })
        .collect::<Vec<_>>()
        .join("  ")
}

// ============================================================
// BUSCA
// ============================================================

fn otimiza<T: Ord + Copy + Send + Sync>(rotulo: &str, base: &[T]) {
    let mut v: Vec<T> = base.to_vec();
    let runs = api::detect_global_trend(base).len();

    println!("\n{}", "=".repeat(78));
    println!("{rotulo}   n={}  runs={}  run_med={}", base.len(), runs, base.len() / runs.max(1));
    println!("{}", "=".repeat(78));

    // ---- referencia e piso de ruido ----
    let multi = mede(base, &mut v, None);
    let d1 = mede(base, &mut v, Some(PimConfig::default()));
    let d2 = mede(base, &mut v, Some(PimConfig::default()));
    let ruido = 100.0 * (d2 - d1).abs() / d1.min(d2);
    let partida = d1.min(d2);

    println!("\nmultimerge          {:>9.1}ms", multi);
    println!("pim default         {:>9.1}ms  ({:+.1}% vs multi)", partida, 100.0*(partida-multi)/multi);
    println!("piso de ruido       {:>9.1}%   (mesma config medida duas vezes)", ruido);
    if ruido > 15.0 {
        println!("\nAVISO: ruido acima de 15%. A busca pode fixar valores por acaso.");
    }
    let margem = ruido.max(3.0);

    // ---- FASE 1: cada variavel isolada ----
    println!("\n--- FASE 1: cada variavel isolada, demais no default ---");
    let mut melhores: Vec<usize> = Vec::new();
    for var in VARS {
        let mut best = (f64::MAX, var.valores[0]);
        let mut linha = String::new();
        for &x in var.valores {
            let mut c = PimConfig::default();
            (var.aplica)(&mut c, x);
            let t = mede(base, &mut v, Some(c));
            linha.push_str(&format!("{}={:.0}  ", (var.mostra)(x), t));
            if t < best.0 { best = (t, x); }
        }
        let ganho = 100.0 * (best.0 - partida) / partida;
        let vale = if ganho < -margem { "*" } else { " " };
        println!("  {}{:<20} melhor={:<8} {:>8.1}ms {:>+7.1}%   [{}]",
                 vale, var.nome, (var.mostra)(best.1), best.0, ganho, linha.trim_end());
        melhores.push(best.1);
    }
    println!("  * = ganho acima do piso de ruido");

    // ---- FASE 2: combina os melhores ----
    let mut comb = PimConfig::default();
    for (i, var) in VARS.iter().enumerate() { (var.aplica)(&mut comb, melhores[i]); }
    let t_comb = mede(base, &mut v, Some(comb));
    println!("\n--- FASE 2: todos os melhores juntos ---");
    println!("  {:>9.1}ms  ({:+.1}% vs default, {:+.1}% vs multi)",
             t_comb, 100.0*(t_comb-partida)/partida, 100.0*(t_comb-multi)/multi);
    println!("  {}", descreve(&comb));
    if t_comb > partida * (1.0 + margem/100.0) {
        println!("  A combinacao ficou PIOR que o default: as variaveis interagem.");
    }

    // ---- FASE 3: descida coordenada a partir da combinacao ----
    println!("\n--- FASE 3: descida coordenada (2 passadas) ---");
    let mut atual = comb;
    let mut t_atual = t_comb;
    for passada in 1..=2 {
        let mut mudou = false;
        for var in VARS {
            let mut best = (t_atual, None);
            for &x in var.valores {
                let mut c = atual;
                (var.aplica)(&mut c, x);
                let t = mede(base, &mut v, Some(c));
                if t < best.0 * (1.0 - margem/100.0) { best = (t, Some(x)); }
            }
            if let Some(x) = best.1 {
                (var.aplica)(&mut atual, x);
                println!("  passada {passada}: {} -> {}  ({:.1}ms, {:+.1}%)",
                         var.nome, (var.mostra)(x), best.0, 100.0*(best.0-t_atual)/t_atual);
                t_atual = best.0;
                mudou = true;
            }
        }
        if !mudou { println!("  passada {passada}: nenhuma melhora acima do ruido"); break; }
    }

    println!("\n--- RESULTADO ---");
    println!("  multimerge      {:>9.1}ms", multi);
    println!("  pim default     {:>9.1}ms  {:+.1}% vs multi", partida, 100.0*(partida-multi)/multi);
    println!("  pim otimizado   {:>9.1}ms  {:+.1}% vs multi  ({:+.1}% vs default)",
             t_atual, 100.0*(t_atual-multi)/multi, 100.0*(t_atual-partida)/partida);
    println!("  config: {}", descreve(&atual));
}

// ============================================================
// DADOS
// ============================================================

fn dicionario() -> &'static [String] {
    let v: Vec<String> = (0..DIC).map(|i| format!("PIM-CHAVE-{i:08X}")).collect();
    Box::leak(v.into_boxed_slice())
}

fn inverso_rep(n: usize, rep: usize, dic: &'static [String]) -> Vec<&'static str> {
    let mut v = Vec::with_capacity(n);
    let mut i = dic.len();
    while v.len() < n && i > 0 {
        i -= 1;
        for _ in 0..rep { if v.len() < n { v.push(dic[i].as_str()); } }
    }
    while v.len() < n { v.push(dic[0].as_str()); }
    v
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
    let dic = dicionario();

    println!("BUSCA AUTO-OTIMIZADA — {} threads | mediana de {REPS} por ponto",
             rayon::current_num_threads());
    println!("Fase 1 varre cada variavel isolada. Fase 2 combina os melhores.");
    println!("Fase 3 refina em descida coordenada, aceitando so ganhos acima do ruido.");

    // o caso que regride: run_med 9
    otimiza("&str INVERSO rep=9 (o caso que perde)", &inverso_rep(n, 9, dic));

    // controle: mesmo padrao, run_med 29, que NAO perde
    otimiza("&str INVERSO rep=30 (controle, nao perde)", &inverso_rep(n, 30, dic));

    println!("\n{}", "=".repeat(78));
    println!("Se a config otima dos dois casos for DIFERENTE, nenhum valor fixo serve");
    println!("e a decisao precisa ser dinamica. Se for a MESMA, e so trocar o default.");
}