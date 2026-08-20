use adaptive_parallel_insertion_merge as api;
use api::cut::{corte_diagonal, merge_back, merge_front, pway_merge_frente};
use api::pim_kernel::{pim_back, pim_front};
use rayon::prelude::*;
use std::time::Instant;

const REPS: usize = 9;

// ============================================================
// BIDIRECIONAL FATIADO
//
// Divide a saida em `threads/2` blocos, corta cada fronteira com
// `corte_diagonal`, e entrega cada bloco a DUAS threads que o mergeiam
// bidirecionalmente -- uma da frente, outra de tras.
//
// CORTES: threads/2 - 1 (as fronteiras interiores). O P-vias com o mesmo
// numero de tarefas paga threads-1. Com 8 threads: 3 contra 7.
//
// Dentro de cada bloco nao ha corte: as entradas ja estao limitadas pelas
// fronteiras, e sobre esse submulticonjunto o teorema vale igual -- a frente
// com empate->A emite os k menores, a tras com empate->B emite o resto.
// ============================================================
fn bidir_fatiado<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    threads: usize,
    galope: bool,
) {
    let blocos = (threads / 2).max(1);
    let total = dest.len();

    if blocos == 1 || total < blocos * 128 {
        let k = total / 2;
        let (df, db) = dest.split_at_mut(k);
        if galope {
            rayon::join(|| pim_front(a, b, df), || pim_back(a, b, db));
        } else {
            rayon::join(|| merge_front(a, b, df), || merge_back(a, b, db));
        }
        return;
    }

    let fr: Vec<usize> = (0..=blocos).map(|i| total * i / blocos).collect();
    let cortes: Vec<(usize, usize)> = fr
        .par_iter()
        .map(|&k| {
            if k == 0 {
                (0, 0)
            } else if k == total {
                (a.len(), b.len())
            } else {
                corte_diagonal(k, a, b)
            }
        })
        .collect();

    let mut pedacos: Vec<&mut [T]> = Vec::with_capacity(blocos);
    let mut resto: &mut [T] = dest;
    for i in 0..blocos - 1 {
        let (esq, dir) = resto.split_at_mut(fr[i + 1] - fr[i]);
        pedacos.push(esq);
        resto = dir;
    }
    pedacos.push(resto);

    pedacos.into_par_iter().enumerate().for_each(|(i, d)| {
        let (ia, ib) = cortes[i];
        let (ja, jb) = cortes[i + 1];
        let sa = &a[ia..ja];
        let sb = &b[ib..jb];
        let k = d.len() / 2;
        let (df, db) = d.split_at_mut(k);
        if galope {
            rayon::join(|| pim_front(sa, sb, df), || pim_back(sa, sb, db));
        } else {
            rayon::join(|| merge_front(sa, sb, df), || merge_back(sa, sb, db));
        }
    });
}

fn bidir_puro<T: Ord + Copy + Send + Sync>(a: &[T], b: &[T], dest: &mut [T], galope: bool) {
    let k = dest.len() / 2;
    let (df, db) = dest.split_at_mut(k);
    if galope {
        rayon::join(|| pim_front(a, b, df), || pim_back(a, b, db));
    } else {
        rayon::join(|| merge_front(a, b, df), || merge_back(a, b, db));
    }
}

// ============================================================
fn mediana(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[derive(Clone, Copy)]
enum M {
    Puro(bool),
    Fatiado(bool),
    Pway(bool),
}

fn nome(m: M, t: usize) -> String {
    match m {
        M::Puro(g) => format!("bidir puro{}", if g { " +gal" } else { "" }),
        M::Fatiado(g) => format!("bidir fatiado {}x2{}", t / 2, if g { " +gal" } else { "" }),
        M::Pway(g) => format!("P-vias P={}{}", t, if g { " +gal" } else { "" }),
    }
}

fn cortes(m: M, t: usize) -> usize {
    match m {
        M::Puro(_) => 0,
        M::Fatiado(_) => (t / 2).max(1) - 1,
        M::Pway(_) => t - 1,
    }
}

fn tabela<T: Ord + Copy + Send + Sync + std::fmt::Debug>(rotulo: &str, ordenado: &[T], c: usize) {
    let n = ordenado.len();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let (mut i, mut pa) = (0usize, true);
    while i < n {
        let f = (i + c).min(n);
        if pa {
            a.extend_from_slice(&ordenado[i..f])
        } else {
            b.extend_from_slice(&ordenado[i..f])
        }
        pa = !pa;
        i = f;
    }
    let mut d: Vec<T> = vec![a[0]; n];

    println!(
        "\n  {rotulo} — {}",
        if c >= n / 2 {
            "faixas disjuntas".to_string()
        } else if c == 1 {
            "intercalacao perfeita".to_string()
        } else {
            format!("blocos de {c}")
        }
    );
    println!(
        "  {:>4} {:>11} {:>13} {:>11} {:>11} {:>13} {:>11}   {}",
        "thr", "puro", "fatiado", "P-vias", "puro+gal", "fatiado+gal", "P-vias+gal", "melhor"
    );
    println!("  {}", "-".repeat(105));

    for &t in &[2usize, 4, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        let ms = [
            M::Puro(false),
            M::Fatiado(false),
            M::Pway(false),
            M::Puro(true),
            M::Fatiado(true),
            M::Pway(true),
        ];
        let mut acc: Vec<Vec<f64>> = vec![vec![]; ms.len()];
        pool.install(|| {
            for rep in 0..REPS + 1 {
                for slot in 0..ms.len() {
                    let id = (rep + slot) % ms.len();
                    let ini = Instant::now();
                    match ms[id] {
                        M::Puro(g) => bidir_puro(&a, &b, &mut d, g),
                        M::Fatiado(g) => bidir_fatiado(&a, &b, &mut d, t, g),
                        M::Pway(g) => pway_merge_frente(&a, &b, &mut d, t, g),
                    }
                    let e = ini.elapsed().as_secs_f64() * 1e3;
                    assert!(
                        d.windows(2).all(|w| w[0] <= w[1]),
                        "{} nao ordenou",
                        nome(ms[id], t)
                    );
                    if rep > 0 {
                        acc[id].push(e);
                    }
                }
            }
        });
        let m: Vec<f64> = acc.iter().map(|x| mediana(&mut x.clone())).collect();
        let (mut bi, mut bv) = (0usize, f64::MAX);
        for (i, &x) in m.iter().enumerate() {
            if x < bv {
                bv = x;
                bi = i;
            }
        }
        print!("  {:>4}", t);
        for x in &m {
            print!("{:>11.2}", x);
        }
        println!("   {}", nome(ms[bi], t));
    }
    println!(
        "  cortes por chamada: puro=0  fatiado={}  P-vias={}   (com 8 threads)",
        cortes(M::Fatiado(false), 8),
        cortes(M::Pway(false), 8)
    );
}

fn main() {
    let m: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000);
    let n = m * 2;
    let mut x = 0x243F6A8885A308D3u64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };

    println!("BIDIRECIONAL FATIADO x P-VIAS");
    println!("duas listas de {m} elementos ordenados | mediana de {REPS} | nucleos: {}",
             std::thread::available_parallelism().map(usize::from).unwrap_or(0));
    println!("\nfatiado: divide a saida em thr/2 blocos, corta as fronteiras com");
    println!("corte_diagonal, e da 2 threads por bloco (frente + tras, sem corte interno).");
    println!("P-vias: divide em thr segmentos, todos para frente, um corte por fronteira.");

    let mut u: Vec<u64> = (0..n).map(|_| rnd()).collect();
    u.sort_unstable();
    println!("\n============ u64 (BANDA) ============");
    for c in [1usize, 1000, m] {
        tabela("u64", &u, c);
    }

    let arena: Vec<String> = (0..n)
        .map(|_| {
            let mut s = String::with_capacity(32);
            for _ in 0..28 {
                s.push('A');
            }
            for _ in 0..4 {
                s.push((b'a' + (rnd() % 26) as u8) as char);
            }
            s
        })
        .collect();
    let arena: &'static [String] = Box::leak(arena.into_boxed_slice());
    let mut st: Vec<&'static str> = arena.iter().map(|s| s.as_str()).collect();
    st.sort_unstable();
    println!("\n\n======= &str (LATENCIA) =======");
    for c in [1usize, 1000, m] {
        tabela("&str", &st, c);
    }

    println!("\nO fatiado paga menos da metade dos cortes do P-vias para o mesmo numero");
    println!("de tarefas. Se empatar em tempo, a economia de cortes nao importa; se");
    println!("ganhar, o custo do particionamento e maior do que as medicoes sugeriam.");
}
