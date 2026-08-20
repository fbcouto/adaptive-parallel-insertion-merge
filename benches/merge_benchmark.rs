//! Bateria de MERGE, nao de sort.
//!
//! Mede a operacao de intercalar DUAS LISTAS JA ORDENADAS -- append de lote em
//! tabela ordenada, uniao de indices, merge de particao, join ordenado.
//!
//! Existe separada porque o ganho da correcao de slices limitados (6.6x medido
//! em faixas disjuntas) NAO aparece num benchmark de sort: ali a estrutura
//! disjunta se dissolve conforme a arvore de merge sobe, e os runs
//! intermediarios ja sao misturas de varios lotes.
//!
//!   cargo bench --bench merge_benchmark
//!   cargo bench --bench merge_benchmark -- "disjuntas/10M"

use adaptive_parallel_insertion_merge::cut::{
    merge_back, merge_front, pway_merge_frente, pway_merge_frente_adaptativo,
};
use adaptive_parallel_insertion_merge::pim_kernel::{merge_ro, pim_back, pim_front};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};

const AMOSTRAS: usize = 10;
const TAMANHOS: &[(usize, &str)] = &[(1_000_000, "1M"), (10_000_000, "10M")];
const DICIONARIO: usize = 1_048_576;

/// Reparte uma sequencia ordenada em duas listas alternando blocos de `bloco`.
///
///   bloco = 1        intercalacao perfeita, cada merge compara em toda posicao
///   bloco = 1000     levemente intercalado
///   bloco = metade   faixas disjuntas, A inteiro antes de B
fn reparte<T: Copy>(ordenado: &[T], bloco: usize) -> (Vec<T>, Vec<T>) {
    let n = ordenado.len();
    let (mut a, mut b) = (Vec::with_capacity(n / 2 + bloco), Vec::with_capacity(n / 2 + bloco));
    let (mut i, mut para_a) = (0usize, true);
    while i < n {
        let fim = (i + bloco).min(n);
        if para_a {
            a.extend_from_slice(&ordenado[i..fim]);
        } else {
            b.extend_from_slice(&ordenado[i..fim]);
        }
        para_a = !para_a;
        i = fim;
    }
    (a, b)
}

fn bidirecional<T: Ord + Copy + Send + Sync>(a: &[T], b: &[T], dest: &mut [T], galope: bool) {
    let k = dest.len() / 2;
    let (frente, tras) = dest.split_at_mut(k);
    if galope {
        rayon::join(|| pim_front(a, b, frente), || pim_back(a, b, tras));
    } else {
        rayon::join(|| merge_front(a, b, frente), || merge_back(a, b, tras));
    }
}

fn mede_grupo<T: Ord + Copy + Send + Sync + 'static>(
    c: &mut Criterion,
    cenario: &str,
    rotulo: &str,
    a: &[T],
    b: &[T],
) {
    let n = a.len() + b.len();
    let modelo = a[0];
    let threads = rayon::current_num_threads();
    let mut grupo = c.benchmark_group(format!("Merge/{cenario}/{rotulo}"));
    grupo.sample_size(AMOSTRAS);

    grupo.bench_function("sequencial", |bch| {
        bch.iter_batched_ref(
            || vec![modelo; n],
            |d| merge_ro(black_box(a), black_box(b), d),
            BatchSize::LargeInput,
        )
    });
    grupo.bench_function("bidirecional", |bch| {
        bch.iter_batched_ref(
            || vec![modelo; n],
            |d| bidirecional(black_box(a), black_box(b), d, false),
            BatchSize::LargeInput,
        )
    });
    grupo.bench_function("bidirecional galopante", |bch| {
        bch.iter_batched_ref(
            || vec![modelo; n],
            |d| bidirecional(black_box(a), black_box(b), d, true),
            BatchSize::LargeInput,
        )
    });
    grupo.bench_function("P-vias", |bch| {
        bch.iter_batched_ref(
            || vec![modelo; n],
            |d| pway_merge_frente(black_box(a), black_box(b), d, threads, false),
            BatchSize::LargeInput,
        )
    });
    grupo.bench_function("P-vias galopante", |bch| {
        bch.iter_batched_ref(
            || vec![modelo; n],
            |d| pway_merge_frente(black_box(a), black_box(b), d, threads, true),
            BatchSize::LargeInput,
        )
    });
    grupo.bench_function("P-vias adaptativo", |bch| {
        bch.iter_batched_ref(
            || vec![modelo; n],
            |d| pway_merge_frente_adaptativo(black_box(a), black_box(b), d, threads, 24),
            BatchSize::LargeInput,
        )
    });
    grupo.finish();
}

fn dicionario() -> &'static [String] {
    let v: Vec<String> = (0..DICIONARIO).map(|i| format!("PIM-CHAVE-{i:08X}")).collect();
    Box::leak(v.into_boxed_slice())
}

fn bench_merge(c: &mut Criterion) {
    let arena = dicionario();
    let dic: Vec<&'static str> = arena.iter().map(String::as_str).collect();

    for &(tamanho, rotulo) in TAMANHOS {
        // ---------- u64 ----------
        let mut rng = StdRng::seed_from_u64(0x4D45_5247);
        let mut u: Vec<u64> = (0..tamanho).map(|_| rng.gen()).collect();
        u.sort_unstable();

        for (bloco, nome) in [
            (1usize, "u64 intercalado"),
            (1_000, "u64 blocos de 1000"),
            (tamanho / 2, "u64 disjuntas"),
        ] {
            let (a, b) = reparte(&u, bloco);
            mede_grupo(c, nome, rotulo, &a, &b);
        }
        drop(u);

        // ---------- &str ----------
        let mut rng = StdRng::seed_from_u64(0x5354_5247);
        let mut s: Vec<&'static str> = (0..tamanho).map(|_| dic[rng.gen_range(0..dic.len())]).collect();
        s.sort_unstable();

        for (bloco, nome) in [
            (1usize, "str intercalado"),
            (1_000, "str blocos de 1000"),
            (tamanho / 2, "str disjuntas"),
        ] {
            let (a, b) = reparte(&s, bloco);
            mede_grupo(c, nome, rotulo, &a, &b);
        }
    }
}

criterion_group!(benches, bench_merge);
criterion_main!(benches);
