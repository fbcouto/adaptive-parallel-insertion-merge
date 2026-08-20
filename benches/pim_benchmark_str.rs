//! Bateria Criterion textual: Rayon, Multimerge e PIM refinado.
//!
//! Usa `&str`, que preserva comparacoes textuais e atende ao requisito `Copy`
//! do PIM. As chaves vivem em um dicionario de 1.048.576 strings; somente os
//! vetores de referencias sao clonados entre as amostras, fora da medicao.
//!
//! Execucao completa:
//!   cargo bench --bench pim_benchmark_str
//!
//! Para igualar os trabalhadores:
//!   $env:RAYON_NUM_THREADS = 8; $env:PIM_NUM_THREADS = 8
//!
//! Para construir somente um caso (economiza memoria em 100M):
//!   $env:PIM_BENCH_CASE = 'Baixa cardinalidade/100M'

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
const TAMANHO_DICIONARIO: usize = 1_048_576;
const CARDINALIDADE_BAIXA: usize = 32;

fn cria_dicionario() -> Vec<String> {
    // Prefixo comum torna cada `Ord` de &str uma comparacao textual real, e a
    // cauda hexadecimal mantem a ordem lexicografica igual a ordem numerica.
    (0..TAMANHO_DICIONARIO)
        .map(|indice| format!("PIM-CHAVE-{indice:08X}"))
        .collect()
}

fn gera_aleatorio<'a>(tamanho: usize, dicionario: &[&'a str]) -> Vec<&'a str> {
    let mut rng = StdRng::seed_from_u64(0x57A1_1E00);
    (0..tamanho)
        .map(|_| dicionario[rng.gen_range(0..dicionario.len())])
        .collect()
}

fn anexa_crescente<'a>(saida: &mut Vec<&'a str>, quantidade: usize, dicionario: &[&'a str]) {
    let por_chave = quantidade / dicionario.len();
    let resto = quantidade % dicionario.len();
    for (indice, &chave) in dicionario.iter().enumerate() {
        let repeticoes = por_chave + usize::from(indice < resto);
        for _ in 0..repeticoes {
            saida.push(chave);
        }
    }
}

fn gera_ordenado<'a>(tamanho: usize, dicionario: &[&'a str]) -> Vec<&'a str> {
    let mut saida = Vec::with_capacity(tamanho);
    anexa_crescente(&mut saida, tamanho, dicionario);
    saida
}

fn gera_inverso<'a>(tamanho: usize, dicionario: &[&'a str]) -> Vec<&'a str> {
    let mut saida = Vec::with_capacity(tamanho);
    let por_chave = tamanho / dicionario.len();
    let resto = tamanho % dicionario.len();
    for (indice, &chave) in dicionario.iter().enumerate().rev() {
        let repeticoes = por_chave + usize::from(indice < resto);
        for _ in 0..repeticoes {
            saida.push(chave);
        }
    }
    saida
}

fn gera_sawtooth<'a>(tamanho: usize, dicionario: &[&'a str]) -> Vec<&'a str> {
    const DENTES: usize = 50;
    let mut saida = Vec::with_capacity(tamanho);
    let por_dente = tamanho / DENTES;
    for _ in 0..DENTES {
        anexa_crescente(&mut saida, por_dente, dicionario);
    }
    let resto = tamanho - saida.len();
    anexa_crescente(&mut saida, resto, dicionario);
    saida
}

fn gera_baixa_cardinalidade<'a>(tamanho: usize, dicionario: &[&'a str]) -> Vec<&'a str> {
    let mut rng = StdRng::seed_from_u64(0x10CA_4D1A);
    (0..tamanho)
        .map(|_| dicionario[rng.gen_range(0..CARDINALIDADE_BAIXA)])
        .collect()
}

fn mede_grupo(c: &mut Criterion, cenario: &str, rotulo: &str, base: &[&str]) {
    let mut grupo = c.benchmark_group(format!("String/{cenario}/{rotulo}"));
    grupo.sample_size(AMOSTRAS);

    grupo.bench_function("Rayon par_sort", |b| {
        b.iter_batched_ref(|| base.to_vec(), |dados| dados.par_sort(), BatchSize::LargeInput)
    });
    grupo.bench_function("Multimerge", |b| {
        b.iter_batched_ref(
            || base.to_vec(),
            |dados| multi_merge_sort(black_box(dados)),
            BatchSize::LargeInput,
        )
    });
    grupo.bench_function("PIM refinado", |b| {
        b.iter_batched_ref(
            || base.to_vec(),
            |dados| pim_sort(black_box(dados)),
            BatchSize::LargeInput,
        )
    });
    grupo.finish();
}

fn inclui_caso(cenario: &str, rotulo: &str, filtro: Option<&str>) -> bool {
    let Some(filtro) = filtro else {
        return true;
    };
    let identificador = format!("{cenario}/{rotulo}")
        .replace(" (32 chaves)", "")
        .to_lowercase();
    identificador.contains(&filtro.to_lowercase())
}

fn bench_sorts(c: &mut Criterion) {
    let arena = cria_dicionario();
    let dicionario: Vec<&str> = arena.iter().map(String::as_str).collect();
    let filtro = std::env::var("PIM_BENCH_CASE").ok();

    for &(tamanho, rotulo) in TAMANHOS {
        // Cada base e descartada antes de construir a proxima. Em 100M, isso
        // evita manter cinco vetores de referencias de 1,6 GB simultaneamente.
        if inclui_caso("Aleatorio", rotulo, filtro.as_deref()) {
            let base = gera_aleatorio(tamanho, &dicionario);
            mede_grupo(c, "Aleatorio", rotulo, &base);
        }
        if inclui_caso("Ordenado", rotulo, filtro.as_deref()) {
            let base = gera_ordenado(tamanho, &dicionario);
            mede_grupo(c, "Ordenado", rotulo, &base);
        }
        if inclui_caso("Inverso", rotulo, filtro.as_deref()) {
            let base = gera_inverso(tamanho, &dicionario);
            mede_grupo(c, "Inverso", rotulo, &base);
        }
        if inclui_caso("Sawtooth", rotulo, filtro.as_deref()) {
            let base = gera_sawtooth(tamanho, &dicionario);
            mede_grupo(c, "Sawtooth", rotulo, &base);
        }
        if inclui_caso("Baixa cardinalidade (32 chaves)", rotulo, filtro.as_deref()) {
            let base = gera_baixa_cardinalidade(tamanho, &dicionario);
            mede_grupo(c, "Baixa cardinalidade (32 chaves)", rotulo, &base);
        }
    }
}

criterion_group!(benches, bench_sorts);
criterion_main!(benches);
