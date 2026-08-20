use adaptive_parallel_insertion_merge as api;
use api::cut::{corte_diagonal, merge_back, merge_front, pway_merge_frente};
use api::pim_kernel::{pim_back, pim_front};
use rayon::prelude::*;
use std::time::Instant;

const REPS: usize = 9;

// ============================================================
// O NUCLEO DA VARIANTE
//
// `pway_merge_frente` passa SUFIXOS: &a[ia..] e &b[ib..], ambos ate o fim.
// Um segmento que caia inteiro dentro de A ainda recebe todo o restante de B,
// e o kernel compara em cada posicao.
//
// Aqui os slices sao LIMITADOS nos dois lados: &a[ia..ja] e &b[ib..jb], onde
// (ja, jb) vem do corte da fronteira SEGUINTE. Quando o bloco cai inteiro em
// um dos lados, o outro slice fica vazio de verdade.
//
// `memcpy_explicito` fecha o circulo: com um lado vazio, copia direto em vez
// de rodar o laco de comparacao. O pim_front ja faz isso internamente; o
// merge_front linear NAO faz.
// ============================================================

#[inline]
fn bloco<T: Ord + Copy + Send + Sync>(
    sa: &[T],
    sb: &[T],
    d: &mut [T],
    galope: bool,
    memcpy: bool,
) {
    if memcpy {
        if sb.is_empty() {
            d.copy_from_slice(&sa[..d.len()]);
            return;
        }
        if sa.is_empty() {
            d.copy_from_slice(&sb[..d.len()]);
            return;
        }
    }
    if galope {
        pim_front(sa, sb, d)
    } else {
        merge_front(sa, sb, d)
    }
}

fn fronteiras<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    total: usize,
    p: usize,
) -> (Vec<usize>, Vec<(usize, usize)>) {
    let fr: Vec<usize> = (0..=p).map(|i| total * i / p).collect();
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
    (fr, cortes)
}

fn fatia<'d, T>(dest: &'d mut [T], fr: &[usize], p: usize) -> Vec<&'d mut [T]> {
    let mut out = Vec::with_capacity(p);
    let mut resto: &mut [T] = dest;
    for i in 0..p - 1 {
        let (esq, dir) = resto.split_at_mut(fr[i + 1] - fr[i]);
        out.push(esq);
        resto = dir;
    }
    out.push(resto);
    out
}

/// P-vias com slices LIMITADOS nos dois lados.
fn pway_limitado<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    p: usize,
    galope: bool,
    memcpy: bool,
) {
    let total = dest.len();
    if p < 2 || total < p * 64 {
        bloco(a, b, dest, galope, memcpy);
        return;
    }
    let (fr, cortes) = fronteiras(a, b, total, p);
    fatia(dest, &fr, p)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, d)| {
            let (ia, ib) = cortes[i];
            let (ja, jb) = cortes[i + 1];
            bloco(&a[ia..ja], &b[ib..jb], d, galope, memcpy);
        });
}

/// Bidirecional fatiado: thr/2 blocos limitados, 2 threads por bloco.
fn fatiado<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    threads: usize,
    galope: bool,
    memcpy: bool,
) {
    let blocos = (threads / 2).max(1);
    let total = dest.len();
    if blocos < 2 || total < blocos * 128 {
        let k = total / 2;
        let (df, db) = dest.split_at_mut(k);
        if galope {
            rayon::join(|| pim_front(a, b, df), || pim_back(a, b, db));
        } else {
            rayon::join(|| merge_front(a, b, df), || merge_back(a, b, db));
        }
        return;
    }
    let (fr, cortes) = fronteiras(a, b, total, blocos);
    fatia(dest, &fr, blocos)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, d)| {
            let (ia, ib) = cortes[i];
            let (ja, jb) = cortes[i + 1];
            let (sa, sb) = (&a[ia..ja], &b[ib..jb]);
            // bloco inteiro de um lado so: nao ha o que mergear
            if memcpy && (sa.is_empty() || sb.is_empty()) {
                let f = if sb.is_empty() { sa } else { sb };
                d.copy_from_slice(&f[..d.len()]);
                return;
            }
            let k = d.len() / 2;
            let (df, db) = d.split_at_mut(k);
            if galope {
                rayon::join(|| pim_front(sa, sb, df), || pim_back(sa, sb, db));
            } else {
                rayon::join(|| merge_front(sa, sb, df), || merge_back(sa, sb, db));
            }
        });
}

// ============================================================
fn mediana(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

const NOMES: [&str; 6] = [
    "pway sufixo",
    "pway limitado",
    "pway lim+cpy",
    "fatiado",
    "fatiado+cpy",
    "pway lim+cpy+gal",
];

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
    print!("  {:>4}", "thr");
    for x in NOMES {
        print!("{:>18}", x);
    }
    println!("   {}", "melhor");
    println!("  {}", "-".repeat(4 + 18 * NOMES.len() + 20));

    for &t in &[2usize, 4, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        let mut acc: Vec<Vec<f64>> = vec![vec![]; NOMES.len()];
        pool.install(|| {
            for rep in 0..REPS + 1 {
                for slot in 0..NOMES.len() {
                    let id = (rep + slot) % NOMES.len();
                    let ini = Instant::now();
                    match id {
                        0 => pway_merge_frente(&a, &b, &mut d, t, false),
                        1 => pway_limitado(&a, &b, &mut d, t, false, false),
                        2 => pway_limitado(&a, &b, &mut d, t, false, true),
                        3 => fatiado(&a, &b, &mut d, t, false, false),
                        4 => fatiado(&a, &b, &mut d, t, false, true),
                        _ => pway_limitado(&a, &b, &mut d, t, true, true),
                    }
                    let e = ini.elapsed().as_secs_f64() * 1e3;
                    assert!(
                        d.windows(2).all(|w| w[0] <= w[1]),
                        "{} nao ordenou",
                        NOMES[id]
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
            print!("{:>18.2}", x);
        }
        println!("   {}", NOMES[bi]);
    }
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

    println!("SLICES LIMITADOS x SUFIXOS — duas listas de {m} | mediana de {REPS}");
    println!("\npway sufixo   = como esta hoje: &a[ia..] e &b[ib..] ate o fim do array");
    println!("pway limitado = &a[ia..ja] e &b[ib..jb], cortados nas DUAS fronteiras");
    println!("+cpy          = com um lado vazio, copy_from_slice em vez do laco");
    println!("fatiado       = thr/2 blocos limitados, 2 threads por bloco");
    println!("\nHIPOTESE: o ganho de 5.5x do fatiado em faixas disjuntas vem de LIMITAR");
    println!("os slices, nao de economizar cortes. Se for isso, 'pway limitado' deve");
    println!("alcancar o fatiado, e 'pway sufixo' deve ser o unico lento.");

    let mut u: Vec<u64> = (0..n).map(|_| rnd()).collect();
    u.sort_unstable();
    println!("\n============ u64 ============");
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
    println!("\n\n======= &str =======");
    for c in [1usize, 1000, m] {
        tabela("&str", &st, c);
    }

    println!("\nSe 'pway limitado' empatar com 'fatiado' em faixas disjuntas, a correcao");
    println!("e trocar os sufixos por slices limitados no pway_merge_frente do cut.rs.");
    println!("Se '+cpy' ganhar de 'limitado', vale tambem o atalho de copia explicita.");
}
