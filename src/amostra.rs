use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Amostra {
    pub taxa_mudanca: f64,

    pub taxa_empate: f64,

    pub comparacoes: usize,

    pub janelas_monotonas: usize,
    pub janelas: usize,
}

impl Amostra {
    pub fn custo_relativo(&self, n: usize) -> f64 {
        if n == 0 { return 0.0; }
        self.comparacoes as f64 / (n as f64 * (n as f64).log2())
    }
}

pub fn amostra_entrada<T: Ord>(arr: &[T], janelas: usize, largura: usize) -> Amostra {
    let disponiveis = arr.len().saturating_sub(1);
    if disponiveis < 8 {
        return Amostra {
            taxa_mudanca: 0.0,
            taxa_empate: 0.0,
            comparacoes: 0,
            janelas_monotonas: 1,
            janelas: 1,
        };
    }
    let janelas = janelas.clamp(1, 64);
    let largura = largura.clamp(8, disponiveis);

    let mut mudancas = 0usize;
    let mut empates = 0usize;
    let mut comparacoes = 0usize;
    let mut monotonas = 0usize;

    for j in 0..janelas {
        let inicio = if janelas == 1 { 0 } else { j * (disponiveis - largura) / (janelas - 1) };
        let fim = inicio + largura;

        let mut ascendente = arr[inicio].cmp(&arr[inicio + 1]) != Ordering::Greater;
        let mut mudou_aqui = false;

        for i in (inicio + 1)..fim {
            let ord = arr[i].cmp(&arr[i + 1]);
            comparacoes += 1;
            if ord == Ordering::Equal {
                empates += 1;
            }
            let atual = ord != Ordering::Greater;
            if atual != ascendente {
                mudancas += 1;
                mudou_aqui = true;
                ascendente = atual;
            }
        }
        if !mudou_aqui {
            monotonas += 1;
        }
    }

    let c = comparacoes.max(1) as f64;
    Amostra {
        taxa_mudanca: mudancas as f64 / c,
        taxa_empate: empates as f64 / c,
        comparacoes,
        janelas_monotonas: monotonas,
        janelas,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rota {
    Caotica,

    RunsLongos,

    RunsCurtos,
}

pub const RUN_MEDIO_MINIMO: usize = 16;

pub fn rota_por_amostra(a: &Amostra, limiar_caos: f64) -> Rota {
    if a.janelas_monotonas == a.janelas {
        return Rota::RunsLongos;
    }
    if a.taxa_mudanca >= limiar_caos {
        return Rota::Caotica;
    }
    Rota::RunsLongos
}

pub fn reclassifica_por_metadata(n: usize, num_runs: usize, palpite: Rota) -> Rota {
    if num_runs <= 1 {
        return Rota::RunsLongos;
    }
    let run_medio = n / num_runs.max(1);
    if run_medio < RUN_MEDIO_MINIMO {
        return Rota::RunsCurtos;
    }
    palpite
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn empate_separa_cardinalidade_de_direcao() {
        let a: Vec<u64> = (0..10_000).collect();
        let s = amostra_entrada(&a, 8, 96);
        assert_eq!(s.taxa_mudanca, 0.0);
        assert_eq!(s.taxa_empate, 0.0);
        assert_eq!(s.janelas_monotonas, s.janelas);

        let b: Vec<u64> = (0..10_000).map(|i| i / 400).collect();
        let s = amostra_entrada(&b, 8, 96);
        assert_eq!(s.taxa_mudanca, 0.0);
        assert!(s.taxa_empate > 0.9, "empate {}", s.taxa_empate);
    }

    #[test]
    fn aleatorio_bate_a_taxa_de_inversao_local() {
        let mut x = 0x243F6A8885A308D3u64;
        let v: Vec<u64> = (0..200_000).map(|_| {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17; x
        }).collect();
        let s = amostra_entrada(&v, 16, 96);
        assert!(s.taxa_mudanca > 0.60 && s.taxa_mudanca < 0.72, "taxa {}", s.taxa_mudanca);
        assert!(s.taxa_empate < 0.01);
    }

    #[test]
    fn metadata_corrige_o_palpite_da_amostra() {
        assert_eq!(
            reclassifica_por_metadata(4_000_000, 457_867, Rota::RunsLongos),
            Rota::RunsCurtos
        );

        assert_eq!(
            reclassifica_por_metadata(4_000_000, 4_008, Rota::RunsLongos),
            Rota::RunsLongos
        );

        assert_eq!(
            reclassifica_por_metadata(1_000_000, 1, Rota::Caotica),
            Rota::RunsLongos
        );
    }

    #[test]
    fn invertido_com_repetidos_engana_a_amostra_mas_nao_o_metadata() {
        let mut v: Vec<u64> = (0..200_000u64).map(|i| i % 500).collect();
        v.sort();
        v.reverse();
        let s = amostra_entrada(&v, 8, 96);

        assert!(s.taxa_mudanca < 0.10, "taxa_mudanca {}", s.taxa_mudanca);

        assert!(s.taxa_empate > 0.90, "taxa_empate {}", s.taxa_empate);
    }
}
