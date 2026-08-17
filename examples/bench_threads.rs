use adaptive_parallel_insertion_merge as api;
use api::cut::{merge_back, merge_front, pway_merge_frente_auto};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const REPS: usize = 7;
const THREADS: [usize; 6] = [1, 2, 4, 8, 16, 32];

fn bidir<T: Ord + Copy + Send + Sync>(a: &[T], b: &[T], d: &mut [T], galope: bool) {
    let k = d.len() / 2;
    let (df, db) = d.split_at_mut(k);
    if galope {
        rayon::join(|| api::pim_kernel::pim_front(a, b, df), || api::pim_kernel::pim_back(a, b, db));
    } else {
        rayon::join(|| merge_front(a, b, df), || merge_back(a, b, db));
    }
}

fn tabela<T: Ord + Copy + Send + Sync + std::fmt::Debug>(rotulo: &str, ordenado: &[T], c: usize) {
    let n = ordenado.len();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let (mut i, mut pa) = (0usize, true);
    while i < n {
        let f = (i + c).min(n);
        if pa { a.extend_from_slice(&ordenado[i..f]) } else { b.extend_from_slice(&ordenado[i..f]) }
        pa = !pa; i = f;
    }
    let mut d: Vec<T> = vec![a[0]; n];

    println!("\n  {rotulo} — {}",
             if c >= n / 2 { "faixas disjuntas".to_string() }
             else if c == 1 { "intercalacao perfeita".to_string() }
             else { format!("blocos de {c}") });
    println!("  {:>4} {:>9} {:>9} {:>9} {:>9}   {:<14} {:>8}",
             "thr", "bidir", "P-vias", "galope", "P+gal", "melhor", "vs bidir");
    println!("  {}", "-".repeat(72));

    for &t in THREADS.iter() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        let mut r = [f64::MAX; 4];
        pool.install(|| {
            for _ in 0..REPS {
                for id in 0..4 {
                    let ini = Instant::now();
                    match id {
                        0 => bidir(&a, &b, &mut d, false),
                        1 => pway_merge_frente_auto(&a, &b, &mut d, t, false),
                        2 => bidir(&a, &b, &mut d, true),
                        _ => pway_merge_frente_auto(&a, &b, &mut d, t, true),
                    }
                    let ms = ini.elapsed().as_secs_f64() * 1e3;
                    assert!(d.windows(2).all(|w| w[0] <= w[1]), "id {id} nao ordenou");
                    if ms < r[id] { r[id] = ms }
                }
            }
        });
        let nomes = ["bidir", "P-vias", "galope", "P+gal"];
        let (mut bi, mut bv) = (0usize, f64::MAX);
        for id in 0..4 { if r[id] < bv { bv = r[id]; bi = id } }
        println!("  {:>4} {:>8.2} {:>8.2} {:>8.2} {:>8.2}   {:<14} {:>7.1}%",
                 t, r[0], r[1], r[2], r[3], nomes[bi], 100.0 * (bv - r[0]) / r[0]);
    }
}

fn main() {
    let m: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(500_000);
    let n = m * 2;
    let mut rr = StdRng::seed_from_u64(7);
    println!("ESCALONAMENTO — {m}+{m} elementos | min de {REPS} | nucleos fisicos: {}",
             std::thread::available_parallelism().map(usize::from).unwrap_or(0));

    let mut u: Vec<u64> = (0..n).map(|_| rr.gen()).collect();
    u.sort_unstable();
    println!("\n============ u64 (BANDA) ============");
    for c in [1usize, 1000, m] { tabela("u64", &u, c); }

    let arena: Vec<String> = (0..n).map(|_| {
        let mut s = String::with_capacity(32);
        for _ in 0..28 { s.push('A'); }
        for _ in 0..4 { s.push(rr.gen_range(b'a'..=b'z') as char); } s
    }).collect();
    let arena: &'static [String] = Box::leak(arena.into_boxed_slice());
    let mut st: Vec<&'static str> = arena.iter().map(|s| s.as_str()).collect();
    st.sort_unstable();
    println!("\n\n======= &str (LATENCIA) =======");
    for c in [1usize, 1000, m] { tabela("&str", &st, c); }

    println!("\n\n======= SORT COMPLETO, dados aleatorios =======");
    println!("  {:>4} {:>11} {:>11} {:>10} {:>10} {:>9}",
             "thr", "par_sort", "pim_sort", "pim vs par", "bloco", "P-vias");
    println!("  {}", "-".repeat(62));
    let base: Vec<u64> = (0..n).map(|_| rr.gen()).collect();
    let mut v = base.clone();
    for &t in THREADS.iter() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        let (mut tp, mut tm) = (f64::MAX, f64::MAX);
        pool.install(|| {
            for _ in 0..REPS {
                v.copy_from_slice(&base);
                let i0 = Instant::now(); v.par_sort();
                let e = i0.elapsed().as_secs_f64() * 1e3; if e < tp { tp = e }
                v.copy_from_slice(&base);
                let i0 = Instant::now(); api::pim_sort(&mut v);
                let e = i0.elapsed().as_secs_f64() * 1e3;
                assert!(v.windows(2).all(|w| w[0] <= w[1]));
                if e < tm { tm = e }
            }
        });
        let bloco = pool.install(|| api::bloco_aleatorio::<u64>(n, rayon::current_num_threads()));
        println!("  {:>4} {:>9.1}ms {:>9.1}ms {:>9.1}% {:>10} {:>9}",
                 t, tp, tm, 100.0 * (tm - tp) / tp, bloco, (t * 4).max(2));
    }
}
