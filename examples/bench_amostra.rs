use adaptive_parallel_insertion_merge as api;
use api::amostra::{amostra_entrada, reclassifica_por_metadata, rota_por_amostra};
use rand::rngs::StdRng; use rand::{Rng, SeedableRng};
use std::time::Instant;

fn custo<T: Ord>(v: &[T], j: usize, w: usize) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..21 {
        let t = Instant::now();
        let a = amostra_entrada(v, j, w);
        let e = t.elapsed().as_secs_f64() * 1e6;
        std::hint::black_box(a.comparacoes);
        if e < best { best = e }
    }
    best
}

fn linha<T: Ord + Copy + Send + Sync>(nome: &str, v: &[T], sort_ms: f64) {
    let a = amostra_entrada(v, 8, 96);
    let runs = api::detect_global_trend(v).len();
    let palpite = rota_por_amostra(&a, 0.18);
    let final_ = reclassifica_por_metadata(v.len(), runs, palpite);
    let us = custo(v, 8, 96);
    println!("  {:<26} mud={:>5.1}%  emp={:>5.1}%  runs={:>9}  run_med={:>7}  {:>11?} -> {:<11?} {:>7.1}us ({:>5.2}% do sort)",
             nome, 100.0*a.taxa_mudanca, 100.0*a.taxa_empate, runs, v.len()/runs.max(1),
             palpite, final_, us, 100.0*us/1000.0/sort_ms);
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);
    let mut r = StdRng::seed_from_u64(7);

    println!("AMOSTRA (8 janelas x 96 comparacoes) — n={n}\n");
    println!("  {:<26} {:>10} {:>11} {:>15} {:>15}", "cenario", "mudanca", "empate", "runs", "reclassificacao");
    println!("  {}", "-".repeat(118));

    let al: Vec<u64> = (0..n).map(|_| r.gen()).collect();
    linha("u64 aleatorio", &al, 30.0);
    let ord: Vec<u64> = (0..n as u64).collect();
    linha("u64 ordenado", &ord, 1.0);
    let inv: Vec<u64> = (0..n as u64).rev().collect();
    linha("u64 inverso", &inv, 4.0);
    let saw: Vec<u64> = (0..n).map(|i| (i % 1000) as u64).collect();
    linha("u64 sawtooth", &saw, 40.0);
    let bc: Vec<u64> = { let mut v: Vec<u64> = (0..n).map(|_| r.gen_range(0..32u64)).collect(); v.sort(); v };
    linha("u64 baixa card. agrupada", &bc, 20.0);

    let arena: Vec<String> = (0..n).map(|_| {
        let mut s = String::with_capacity(32);
        for _ in 0..28 { s.push('A'); }
        for _ in 0..4 { s.push(r.gen_range(b'a'..=b'z') as char); } s
    }).collect();
    let arena: &'static [String] = Box::leak(arena.into_boxed_slice());
    let mut st: Vec<&'static str> = arena.iter().map(|s| s.as_str()).collect();
    linha("&str aleatorio", &st, 250.0);
    st.sort_unstable(); st.reverse();
    linha("&str INVERSO c/ repetidos", &st, 600.0);

    println!("\n  mud = taxa de INVERSAO LOCAL (aleatorio i.i.d. tende a 2/3, nao a 41%)");
    println!("  emp = fracao de empates exatos; alta em cardinalidade baixa OU em dados agrupados");
    println!("  A ultima coluna mostra o palpite da amostra corrigido pelo metadata.");
}
