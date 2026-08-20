//! Comparacao Criterion entre Rayon, Multimerge e o PIM refinado.
//!
//! O PIM preserva seu caminho adaptativo nos dados estruturados. Somente a
//! entrada que o escudo de entropia classifica como aleatoria segue para
//! `sort(1920)` e a arvore P-way global.
//!
//! Execucao completa:
//!   cargo bench --bench pim_benchmark
//!
//! Para uma comparacao com a mesma quantidade de trabalhadores:
//!   $env:RAYON_NUM_THREADS = 8; $env:PIM_NUM_THREADS = 8
//!
//! Um grupo especifico pode ser filtrado, por exemplo:
//!   cargo bench --bench pim_benchmark -- "Aleatorio/10M"

use adaptive_parallel_insertion_merge::multimerge::multi_merge_sort;
use adaptive_parallel_insertion_merge::pim_sort;
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::slice::ParallelSliceMut;

const AMOSTRAS: usize = 10;
const TAMANHOS: &[(usize, &str)] = &[
    (1_000_000, "1M"),
    (10_000_000, "10M"),
    (100_000_000, "100M"),
];

type Gerador = fn(usize) -> Vec<u64>;

fn gera_aleatorio(tamanho: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0xA11E_A701);
    (0..tamanho).map(|_| rng.gen()).collect()
}

fn gera_ordenado(tamanho: usize) -> Vec<u64> {
    (0..tamanho as u64).collect()
}

fn gera_inverso(tamanho: usize) -> Vec<u64> {
    (0..tamanho as u64).rev().collect()
}

fn gera_sawtooth(tamanho: usize) -> Vec<u64> {
    let dentes = 50usize;
    let por_dente = tamanho / dentes;
    let mut dados = Vec::with_capacity(tamanho);
    for _ in 0..dentes {
        dados.extend(0..por_dente as u64);
    }
    dados.extend(0..(tamanho - dados.len()) as u64);
    dados
}

/// Chaves repetidas, mas em ordem aleatoria. O PIM deve continuar estavel e o
/// escudo de entropia deve encaminhar este caso ao caminho aleatorio.
fn gera_baixa_cardinalidade(tamanho: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(0x10CA_4D1A);
    (0..tamanho).map(|_| rng.gen_range(0..32u64)).collect()
}

fn mede_grupo(c: &mut Criterion, cenario: &str, rotulo: &str, tamanho: usize, gera: Gerador) {
    let mut grupo = c.benchmark_group(format!("{cenario}/{rotulo}"));
    grupo.sample_size(AMOSTRAS);

    grupo.bench_function("Rayon par_sort", |b| {
        b.iter_batched_ref(
            || gera(tamanho),
            |dados| dados.par_sort(),
            BatchSize::LargeInput,
        )
    });
    grupo.bench_function("Multimerge", |b| {
        b.iter_batched_ref(
            || gera(tamanho),
            |dados| multi_merge_sort(black_box(dados)),
            BatchSize::LargeInput,
        )
    });
    grupo.bench_function("PIM refinado", |b| {
        b.iter_batched_ref(
            || gera(tamanho),
            |dados| pim_sort(black_box(dados)),
            BatchSize::LargeInput,
        )
    });
    grupo.finish();
}

fn bench_sorts(c: &mut Criterion) {
    let cenarios: &[(&str, Gerador)] = &[
        ("Aleatorio", gera_aleatorio),
        ("Ordenado", gera_ordenado),
        ("Inverso", gera_inverso),
        ("Sawtooth", gera_sawtooth),
        ("Baixa cardinalidade (32 chaves)", gera_baixa_cardinalidade),
    ];

    for &(tamanho, rotulo) in TAMANHOS {
        for &(cenario, gera) in cenarios {
            mede_grupo(c, cenario, rotulo, tamanho, gera);
        }
    }
}

criterion_group!(benches, bench_sorts);
criterion_main!(benches);
