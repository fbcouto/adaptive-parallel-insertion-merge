use adaptive_parallel_insertion_merge as api;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const REPS: usize = 9;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
    let thr = rayon::current_num_threads();
    let mut r = StdRng::seed_from_u64(7);
    let base: Vec<u64> = (0..n).map(|_| r.gen()).collect();
    let mut v = base.clone();

    let mut medir = |rotulo: &str, bloco: usize| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..REPS {
            v.copy_from_slice(&base);
            api::set_pway_block(bloco);
            let t = Instant::now();
            if bloco == usize::MAX { v.par_sort() } else { api::pim_sort(&mut v) }
            let ms = t.elapsed().as_secs_f64() * 1e3;
            assert!(v.windows(2).all(|w| w[0] <= w[1]), "{rotulo} nao ordenou");
            if ms < best { best = ms }
        }
        api::set_pway_block(0);
        best
    };

    api::set_pway_block(0);
    let formula = api::bloco_aleatorio::<u64>(n, thr);

    println!("ALEATORIO — {n} u64 ({:.0} MB) | {thr} threads | min de {REPS}", (n * 8) as f64 / 1e6);
    println!("caminho medido: pim_sort -> escudo -> pim_sort_aleatorio (producao)");
    println!("formula bloco_aleatorio(N={n}, threads={thr}) = {formula}\n");

    let t_par = medir("par_sort", usize::MAX);
    println!("{:<26} {:>10.1}ms {:>10}", "rayon par_sort", t_par, "(referencia)");

    let t_1920 = medir("B=1920", 1_920);
    println!("{:<26} {:>10.1}ms {:>9.1}%   <- valor antigo\n", "pim_sort B=1920", t_1920, 100.0 * (t_1920 - t_par) / t_par);

    println!("{:<26} {:>10} {:>10} {:>10} {:>9}", "bloco", "min", "vs par", "vs 1920", "runs");
    println!("{}", "-".repeat(70));

    let mut blocos: Vec<usize> = vec![1_920, 8_192, 32_768, 87_381, 262_144, 1_048_576, 4_194_304];
    if !blocos.contains(&formula) { blocos.push(formula); }
    blocos.sort_unstable();
    blocos.dedup();
    blocos.retain(|&b| b < n);

    let mut melhor = (f64::MAX, 0usize);
    for b in blocos {
        let t = medir("varredura", b);
        if t < melhor.0 { melhor = (t, b); }
        println!("{:<26} {:>8.1}ms {:>9.1}% {:>9.1}% {:>9}",
                 format!("{}{}", b, if b == formula { " <- formula" } else { "" }),
                 t, 100.0 * (t - t_par) / t_par, 100.0 * (t - t_1920) / t_1920, n.div_ceil(b));
    }

    let t_auto = medir("auto", 0);
    println!("\n{:<26} {:>8.1}ms {:>9.1}% {:>9.1}%", "AUTO (formula ligada)",
             t_auto, 100.0 * (t_auto - t_par) / t_par, 100.0 * (t_auto - t_1920) / t_1920);

    println!("\nMELHOR: bloco {} com {:.1}ms  ({:+.1}% vs par_sort, {:+.1}% vs B=1920)",
             melhor.1, melhor.0,
             100.0 * (melhor.0 - t_par) / t_par, 100.0 * (melhor.0 - t_1920) / t_1920);
    if melhor.1 != formula {
        println!("A formula escolheu {formula}, o otimo medido foi {}. Ajuste set_l3_bytes.", melhor.1);
    }
}
