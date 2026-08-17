use crate::{PimConfig, PimError};

pub trait Chave: Ord + Copy + Send + Sync {
    const COMPARACAO_CARA: bool;
}

#[macro_export]
macro_rules! perfil_chave {
    ($t:ty => barata) => {
        impl $crate::despacho::Chave for $t {
            const COMPARACAO_CARA: bool = false;
        }
    };
    ($t:ty => cara) => {
        impl $crate::despacho::Chave for $t {
            const COMPARACAO_CARA: bool = true;
        }
    };
}

macro_rules! barata {
    ($($t:ty),*) => { $( impl Chave for $t { const COMPARACAO_CARA: bool = false; } )* };
}
barata!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, char, bool);

impl<const N: usize> Chave for [u8; N] {
    const COMPARACAO_CARA: bool = false;
}

impl<'a> Chave for &'a str {
    const COMPARACAO_CARA: bool = true;
}
impl<'a> Chave for &'a [u8] {
    const COMPARACAO_CARA: bool = true;
}

pub fn sort<T: Chave>(arr: &mut [T]) {
    sort_com_config(arr, config_para_perfil(T::COMPARACAO_CARA))
}

pub fn sort_com_perfil<T: Ord + Copy + Send + Sync>(arr: &mut [T], comparacao_cara: bool) {
    sort_com_config(arr, config_para_perfil(comparacao_cara));
}

pub fn sort_com_config<T: Ord + Copy + Send + Sync>(arr: &mut [T], config: PimConfig) {
    crate::pim_sort_with_config(arr, config)
}

pub fn try_sort_com_config<T: Ord + Copy + Send + Sync>(
    arr: &mut [T],
    config: PimConfig,
) -> Result<(), PimError> {
    crate::try_pim_sort_with_config(arr, config)
}

fn config_para_perfil(comparacao_cara: bool) -> PimConfig {
    let config = PimConfig::default();
    if comparacao_cara {
        config.comparacao_indireta()
    } else {
        config
    }
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::multimerge;

    #[derive(Clone, Copy, Debug)]
    struct K {
        k: u32,
        i: u32,
    }
    impl PartialEq for K { fn eq(&self, o: &Self) -> bool { self.k == o.k } }
    impl Eq for K {}
    impl PartialOrd for K { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
    impl Ord for K { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.k.cmp(&o.k) } }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn ate(&mut self, n: u64) -> u64 { if n == 0 { 0 } else { self.next() % n } }
    }

    #[test]
    fn os_dois_perfis_dao_a_mesma_saida_estavel() {
        let mut r = Rng(0x9E3779B97F4A7C15);
        for t in 0..120 {
            let n = 20_000 + (t * 373) as usize;
            let card = [3u32, 60, 900_000][t % 3];
            let mut v: Vec<K> = (0..n as u32).map(|i| K { k: r.ate(card as u64) as u32, i }).collect();
            if t % 2 == 0 {
                v.sort();
                for _ in 0..n / 200 {
                    let (a, b) = (r.ate(n as u64) as usize, r.ate(n as u64) as usize);
                    v.swap(a, b);
                }
            }
            let mut esperado = v.clone();
            esperado.sort();

            let mut barata = v.clone();
            sort_com_perfil(&mut barata, false);
            let mut cara = v.clone();
            sort_com_perfil(&mut cara, true);

            let e: Vec<_> = esperado.iter().map(|x| (x.k, x.i)).collect();
            assert_eq!(barata.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(), e, "perfil barato divergiu, t={t}");
            assert_eq!(cara.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(), e, "perfil caro divergiu, t={t}");
        }
    }

    #[test]
    fn perfis_declarados_batem_com_a_medicao() {
        assert!(!<u64 as Chave>::COMPARACAO_CARA);
        assert!(!<[u8; 32] as Chave>::COMPARACAO_CARA);

        assert!(<&'static str as Chave>::COMPARACAO_CARA);
        assert!(<&'static [u8] as Chave>::COMPARACAO_CARA);
    }

    #[test]
    fn referencias_de_texto_sem_lifetime_static_entram_no_despacho() {
        let arena = [String::from("zeta"), String::from("alfa"), String::from("beta")];
        let mut refs: Vec<&str> = arena.iter().map(String::as_str).collect();
        sort(&mut refs);
        assert_eq!(refs, ["alfa", "beta", "zeta"]);
    }

    #[test]
    fn configuracao_e_restaurada_apos_a_chamada() {
        use std::sync::atomic::Ordering::Relaxed;
        multimerge::set_leaf_floor(4096);
        crate::set_folha_modo(0);
        let mut v: Vec<u64> = (0..50_000u64).rev().collect();
        sort_com_perfil(&mut v, true);
        assert_eq!(multimerge::LEAF_FLOOR.load(Relaxed), 4096);
        assert_eq!(crate::FOLHA_MODO.load(Relaxed), 0);
    }
}
