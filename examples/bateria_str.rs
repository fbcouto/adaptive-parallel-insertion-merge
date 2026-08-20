use adaptive_parallel_insertion_merge as api;
use api::{multimerge, PimConfig};
use std::time::Instant;

const REPS: usize = 9;
const DIC: usize = 1_048_576;

fn dicionario() -> &'static [String] {
    let v: Vec<String> = (0..DIC).map(|i| format!("PIM-CHAVE-{i:08X}")).collect();
    Box::leak(v.into_boxed_slice())
}

fn gera_inverso(n: usize, dic: &'static [String]) -> Vec<&'static str> {
    let por = n / dic.len();
    let resto = n % dic.len();
    let mut v = Vec::with_capacity(n);
    for (i, k) in dic.iter().enumerate().rev() {
        for _ in 0..(por + usize::from(i < resto)) {
            v.push(k.as_str());
        }
    }
    v
}

fn gera_inverso_rep(n: usize, rep: usize, dic: &'static [String]) -> Vec<&'static str> {
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

fn mediana(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn dispersao(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    100.0 * (v[v.len() - 1] - v[0]) / v[0]
}

fn compara(base: &[&'static str], cfgs: &[(String, Option<PimConfig>)]) -> Vec<f64> {
    let mut v: Vec<&'static str> = base.to_vec();
    let mut t: Vec<Vec<f64>> = vec![vec![]; cfgs.len()];
    for rep in 0..REPS + 1 {
        for slot in 0..cfgs.len() {
            let id = (rep + slot) % cfgs.len();
            v.copy_from_slice(base);
            let ini = Instant::now();
            match cfgs[id].1 {
                None => multimerge::multi_merge_sort(&mut v),
                Some(c) => api::pim_sort_with_config(&mut v, c),
            }
            let ms = ini.elapsed().as_secs_f64() * 1e3;
            assert!(v.windows(2).all(|w| w[0] <= w[1]), "{} nao ordenou", cfgs[id].0);
            if rep > 0 {
                t[id].push(ms);
            }
        }
    }
    t.iter().map(|x| mediana(&mut x.clone())).collect()
}

fn cfg(f: impl Fn(&mut PimConfig)) -> Option<PimConfig> {
    let mut c = PimConfig::default();
    f(&mut c);
    Some(c)
}

// ============================================================
fn teste1_leaf(base: &[&'static str]) {
    println!("\n=== TESTE 1 — leaf_size ===");
    println!("HIPOTESE: o PIM usa tamanho_folha_config, que para &str (16 bytes) da");
    println!("32768/16 = 2048. O multimerge usa get_leaf_size, cujo piso e 4096.");
    println!("Com 1 milhao de runs essa diferenca multiplica o numero de folhas.");
    println!("SE FOR ISTO: alguma coluna deve cair para perto de 0%.\n");

    let mut c: Vec<(String, Option<PimConfig>)> = vec![("multimerge".into(), None)];
    for f in [512usize, 1024, 2048, 4096, 8192] {
        c.push((format!("folha={f}"), cfg(|x| x.folha_override = f)));
    }
    let m = compara(base, &c);
    for (i, (n, _)) in c.iter().enumerate() {
        if i == 0 {
            println!("  {:<18} {:>9.1}ms", n, m[0]);
        } else {
            println!("  {:<18} {:>9.1}ms {:>+8.1}%", n, m[i], 100.0 * (m[i] - m[0]) / m[0]);
        }
    }
}

// ============================================================
fn teste2_fases(base: &[&'static str]) {
    println!("\n=== TESTE 2 — decomposicao de fases ===");
    println!("Mede cada etapa ISOLADA. A soma deve chegar perto do tempo total;");
    println!("o que sobrar e overhead que a instrumentacao nao captura.\n");

    let n = base.len();
    let mut v: Vec<&'static str> = base.to_vec();
    let (mut td, mut to, mut tb, mut tm2, mut tm4) = (vec![], vec![], vec![], vec![], vec![]);

    for _ in 0..REPS {
        v.copy_from_slice(base);
        let i0 = Instant::now();
        let meta = api::detect_global_trend(&v);
        td.push(i0.elapsed().as_secs_f64() * 1e3);

        let i0 = Instant::now();
        let off = multimerge::block_offsets(&meta);
        to.push(i0.elapsed().as_secs_f64() * 1e3);

        let i0 = Instant::now();
        let mut buf: Vec<&'static str> = vec![v[0]; n];
        tb.push(i0.elapsed().as_secs_f64() * 1e3);

        v.copy_from_slice(base);
        let i0 = Instant::now();
        multimerge::bottom_up_merge_kway(&mut v, &mut buf, &meta, &off, 2048, false);
        tm2.push(i0.elapsed().as_secs_f64() * 1e3);

        v.copy_from_slice(base);
        let i0 = Instant::now();
        multimerge::bottom_up_merge_kway(&mut v, &mut buf, &meta, &off, 4096, false);
        tm4.push(i0.elapsed().as_secs_f64() * 1e3);
    }

    let c: Vec<(String, Option<PimConfig>)> = vec![
        ("multimerge".into(), None),
        ("pim default".into(), cfg(|_| {})),
    ];
    let m = compara(base, &c);

    println!("  {:<26} {:>10}", "TOTAL multimerge", format!("{:.1}ms", m[0]));
    println!("  {:<26} {:>10} {:>+8.1}%", "TOTAL pim default", format!("{:.1}ms", m[1]),
             100.0 * (m[1] - m[0]) / m[0]);
    println!();
    println!("  {:<26} {:>10}", "detect_global_trend", format!("{:.1}ms", mediana(&mut td)));
    println!("  {:<26} {:>10}", "block_offsets", format!("{:.1}ms", mediana(&mut to)));
    println!("  {:<26} {:>10}", "buffer vec![]", format!("{:.1}ms", mediana(&mut tb)));
    println!("  {:<26} {:>10}", "merge kway folha=2048", format!("{:.1}ms", mediana(&mut tm2)));
    println!("  {:<26} {:>10}", "merge kway folha=4096", format!("{:.1}ms", mediana(&mut tm4)));
    println!();
    println!("  Se merge(2048) e merge(4096) forem iguais, o leaf_size NAO explica.");
    println!("  Se a soma ficar perto do multimerge mas longe do pim, sobra overhead");
    println!("  no caminho do pim que nenhuma destas fases captura.");
}

// ============================================================
fn teste3_run_med(dic: &'static [String], n: usize) {
    println!("\n=== TESTE 3 — onde o gap aparece em funcao do run_med ===");
    println!("Mesmo padrao (descida com platos), variando quantas vezes cada chave repete.");
    println!("rep=1 da uma descida estrita (runs=1, atalho O(N)); rep alto da runs longos.\n");
    println!("  {:>5} {:>9} {:>12} {:>12} {:>9}", "rep", "run_med", "multimerge", "pim", "delta");
    println!("  {}", "-".repeat(52));

    for rep in [1usize, 3, 9, 30, 95, 300] {
        let base = gera_inverso_rep(n, rep, dic);
        let runs = api::detect_global_trend(&base).len();
        let c: Vec<(String, Option<PimConfig>)> = vec![
            ("multi".into(), None),
            ("pim".into(), cfg(|_| {})),
        ];
        let m = compara(&base, &c);
        println!("  {:>5} {:>9} {:>10.1}ms {:>10.1}ms {:>+8.1}%",
                 rep, base.len() / runs.max(1), m[0], m[1], 100.0 * (m[1] - m[0]) / m[0]);
    }
    println!("\n  Se o gap so aparece em alguma faixa de rep, ha limiar. Se e constante,");
    println!("  a causa independe do comprimento de run.");
}

// ============================================================
fn teste4_controle(base: &[&'static str]) {
    println!("\n=== TESTE 4 — controle de ruido ===");
    println!("A mesma configuracao medida duas vezes. A diferenca entre as duas colunas");
    println!("e o piso de ruido: nada abaixo disso nas tabelas acima e legivel.\n");

    let c: Vec<(String, Option<PimConfig>)> = vec![
        ("multimerge A".into(), None),
        ("pim A".into(), cfg(|_| {})),
        ("multimerge B".into(), None),
        ("pim B".into(), cfg(|_| {})),
    ];
    let mut v: Vec<&'static str> = base.to_vec();
    let mut t: Vec<Vec<f64>> = vec![vec![]; c.len()];
    for rep in 0..REPS + 1 {
        for slot in 0..c.len() {
            let id = (rep + slot) % c.len();
            v.copy_from_slice(base);
            let ini = Instant::now();
            match c[id].1 {
                None => multimerge::multi_merge_sort(&mut v),
                Some(k) => api::pim_sort_with_config(&mut v, k),
            }
            let ms = ini.elapsed().as_secs_f64() * 1e3;
            if rep > 0 { t[id].push(ms); }
        }
    }
    for (i, (n, _)) in c.iter().enumerate() {
        println!("  {:<16} mediana={:>8.1}ms   dispersao={:>5.1}%",
                 n, mediana(&mut t[i].clone()), dispersao(&mut t[i].clone()));
    }
    let a = mediana(&mut t[0].clone());
    let b = mediana(&mut t[2].clone());
    println!("\n  multimerge A vs B: {:+.1}%   <- este e o piso de ruido",
             100.0 * (b - a) / a);
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
    let dic = dicionario();
    let base = gera_inverso(n, dic);
    let runs = api::detect_global_trend(&base).len();

    println!("BATERIA — &str INVERSO (gerador do criterion)");
    println!("n={n} | {} threads | mediana de {REPS}, ordem rotativa, buffer reutilizado",
             rayon::current_num_threads());
    println!("runs={}, run_med={}", runs, base.len() / runs.max(1));
    println!("\nJA DESCARTADOS por medicao anterior: roteamento (6 configs, todas +55%),");
    println!("arvore de merge (pway e kway iguais), kernel de folha (galope on/off e");
    println!("limiar 7/64/512, todos iguais), alocacao de buffer.");

    teste4_controle(&base);
    teste1_leaf(&base);
    teste2_fases(&base);
    teste3_run_med(dic, n);
}
