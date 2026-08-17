use rayon::join;
use rayon::prelude::*;

pub fn corte_diagonal<T: Ord>(k: usize, a: &[T], b: &[T]) -> (usize, usize) {
    let (m, n) = (a.len(), b.len());
    debug_assert!(k <= m + n);

    let mut lo = k.saturating_sub(n);
    let mut hi = k.min(m);

    while lo < hi {
        let i = lo + (hi - lo + 1) / 2;
        let j = k - i;
        if j == n || a[i - 1] <= b[j] {
            lo = i;
        } else {
            hi = i - 1;
        }
    }
    (lo, k - lo)
}

pub fn corte_eliminacao<T: Ord + Copy>(k: usize, a: &[T], b: &[T]) -> (usize, usize) {
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut falta = k;

    loop {
        if falta == 0 {
            return (ia, ib);
        }
        if ia == a.len() {
            return (ia, ib + falta);
        }
        if ib == b.len() {
            return (ia + falta, ib);
        }
        if falta == 1 {
            return if a[ia] <= b[ib] { (ia + 1, ib) } else { (ia, ib + 1) };
        }

        let passo = falta / 2;
        let pa = (ia + passo).min(a.len()) - ia;
        let pb = (ib + passo).min(b.len()) - ib;

        if a[ia + pa - 1] <= b[ib + pb - 1] {
            ia += pa;
            falta -= pa;
        } else {
            ib += pb;
            falta -= pb;
        }
    }
}

#[inline]
pub fn merge_front<T: Ord + Copy>(a: &[T], b: &[T], dest: &mut [T]) {
    let (mut i, mut j) = (0usize, 0usize);
    for slot in dest.iter_mut() {
        let pega_a = j >= b.len() || (i < a.len() && a[i] <= b[j]);
        if pega_a {
            *slot = a[i];
            i += 1;
        } else {
            *slot = b[j];
            j += 1;
        }
    }
}

#[inline]
pub fn merge_back<T: Ord + Copy>(a: &[T], b: &[T], dest: &mut [T]) {
    let (mut qa, mut qb) = (a.len(), b.len());
    for slot in dest.iter_mut().rev() {
        let pega_b = qa == 0 || (qb > 0 && b[qb - 1] >= a[qa - 1]);
        if pega_b {
            qb -= 1;
            *slot = b[qb];
        } else {
            qa -= 1;
            *slot = a[qa];
        }
    }
}

pub fn trisect_merge<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    frac_ponta: f64,
) {
    let total = a.len() + b.len();
    debug_assert_eq!(dest.len(), total);

    if total < 3 {
        merge_front(a, b, dest);
        return;
    }

    let f = frac_ponta.clamp(0.05, 0.49);
    let k1 = ((total as f64 * f) as usize).max(1);
    let k2 = (total - k1).max(k1 + 1);

    let (i, j) = corte_diagonal(k1, a, b);

    let (d_frente, resto) = dest.split_at_mut(k1);
    let (d_meio, d_tras) = resto.split_at_mut(k2 - k1);

    join(
        || merge_front(a, b, d_frente),
        || {
            join(

                || merge_front(&a[i..], &b[j..], d_meio),
                || merge_back(a, b, d_tras),
            )
        },
    );
}

#[cfg(test)]
mod testes {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct K {
        k: u32,
        i: u32,
    }
    impl PartialOrd for K {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for K {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.k.cmp(&o.k)
        }
    }

    fn merge_ref(a: &[K], b: &[K]) -> Vec<K> {
        let (mut i, mut j) = (0, 0);
        let mut out = Vec::with_capacity(a.len() + b.len());
        while i < a.len() && j < b.len() {
            if a[i] <= b[j] {
                out.push(a[i]);
                i += 1
            } else {
                out.push(b[j]);
                j += 1
            }
        }
        out.extend_from_slice(&a[i..]);
        out.extend_from_slice(&b[j..]);
        out
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn ate(&mut self, n: u64) -> u64 {
            if n == 0 { 0 } else { self.next() % n }
        }
    }

    fn par(r: &mut Rng, m: usize, n: usize, card: u32) -> (Vec<K>, Vec<K>) {
        let mut idx = 0u32;
        let mut a: Vec<K> = (0..m)
            .map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x })
            .collect();
        let mut b: Vec<K> = (0..n)
            .map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x })
            .collect();
        a.sort();
        b.sort();
        (a, b)
    }

    #[test]
    fn exemplo_do_paper_20_elementos() {
        let a: Vec<u32> = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
        let b: Vec<u32> = vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
        assert_eq!(corte_diagonal(8, &a, &b), (4, 4));
        assert_eq!(corte_eliminacao(8, &a, &b), (4, 4));

        assert_eq!(a[4].min(b[4]), 9);
    }

    #[test]
    fn os_dois_cortes_concordam_e_valem_as_invariantes() {
        let mut r = Rng(0x243F6A8885A308D3);
        for _ in 0..3000 {
            let m = r.ate(60) as usize;
            let n = r.ate(60) as usize;
            let card = [2u32, 6, 30, 5000][r.ate(4) as usize];
            let (a, b) = par(&mut r, m, n, card);
            let esperado = merge_ref(&a, &b);

            for k in 0..=(m + n) {
                let (i, j) = corte_diagonal(k, &a, &b);
                let (i2, j2) = corte_eliminacao(k, &a, &b);
                assert_eq!((i, j), (i2, j2), "diagonal != eliminacao em k={k}");
                assert_eq!(i + j, k);

                if i > 0 && j < n { assert!(a[i - 1] <= b[j]); }
                if j > 0 && i < m { assert!(b[j - 1] < a[i]); }

                let mut pref = a[..i].to_vec();
                pref.extend_from_slice(&b[..j]);
                pref.sort();
                let mut alvo = esperado[..k].to_vec();
                alvo.sort();
                assert_eq!(pref, alvo, "corte errado em k={k}");
            }
        }
    }

    #[test]
    fn trisect_bate_o_merge_estavel_de_referencia() {
        let mut r = Rng(0x13198A2E03707344);
        for t in 0..4000 {
            let m = r.ate(150) as usize;
            let n = r.ate(150) as usize;
            let card = [2u32, 9, 40, 9999][r.ate(4) as usize];
            let (a, b) = par(&mut r, m, n, card);
            let esperado = merge_ref(&a, &b);
            let frac = [0.1f64, 0.25, 0.4, 0.49][r.ate(4) as usize];

            let mut got = vec![K { k: 0, i: 0 }; m + n];
            trisect_merge(&a, &b, &mut got, frac);
            assert_eq!(
                got.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(),
                esperado.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(),
                "trisect divergiu: t={t} m={m} n={n} frac={frac}"
            );
        }
    }
}

pub fn trisect_merge_gallop<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    frac_ponta: f64,
) {
    use crate::pim_kernel::{pim_back, pim_front};
    let total = a.len() + b.len();
    debug_assert_eq!(dest.len(), total);

    if total < 6 {
        pim_front(a, b, dest);
        return;
    }

    let f = frac_ponta.clamp(0.05, 0.49);
    let k1 = ((total as f64 * f) as usize).max(1);
    let k2 = (total - k1).max(k1 + 1);

    let (i, j) = corte_diagonal(k1, a, b);

    let (d_frente, resto) = dest.split_at_mut(k1);
    let (d_meio, d_tras) = resto.split_at_mut(k2 - k1);

    join(
        || pim_front(a, b, d_frente),
        || {
            join(
                || pim_front(&a[i..], &b[j..], d_meio),
                || pim_back(a, b, d_tras),
            )
        },
    );
}

#[cfg(test)]
mod testes_gallop {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct K { k: u32, i: u32 }
    impl PartialOrd for K { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
    impl Ord for K { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.k.cmp(&o.k) } }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn ate(&mut self, n: u64) -> u64 { if n == 0 { 0 } else { self.next() % n } }
    }

    #[test]
    fn trisect_galopante_bate_a_referencia() {
        let mut r = Rng(0xA4093822299F31D0);
        for t in 0..4000 {
            let (m, n) = (r.ate(150) as usize, r.ate(150) as usize);
            let card = [2u32, 9, 40, 9999][r.ate(4) as usize];
            let mut idx = 0u32;
            let mut a: Vec<K> = (0..m).map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x }).collect();
            let mut b: Vec<K> = (0..n).map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x }).collect();
            a.sort(); b.sort();

            let (mut ii, mut jj) = (0, 0);
            let mut esperado = Vec::with_capacity(m + n);
            while ii < m && jj < n {
                if a[ii] <= b[jj] { esperado.push(a[ii]); ii += 1 } else { esperado.push(b[jj]); jj += 1 }
            }
            esperado.extend_from_slice(&a[ii..]);
            esperado.extend_from_slice(&b[jj..]);

            let frac = [0.1f64, 0.25, 0.33, 0.4, 0.49][r.ate(5) as usize];
            let mut got = vec![K { k: 0, i: 0 }; m + n];
            trisect_merge_gallop(&a, &b, &mut got, frac);
            assert_eq!(
                got.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(),
                esperado.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(),
                "t={t} m={m} n={n} frac={frac}"
            );
        }
    }
}

pub fn pway_merge<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    p: usize,
    frac_ponta: f64,
) {
    let total = a.len() + b.len();
    debug_assert_eq!(dest.len(), total);

    if p < 3 || total < p * 64 {
        let k = total / 2;
        let (df, db) = dest.split_at_mut(k);
        rayon::join(|| merge_front(a, b, df), || merge_back(a, b, db));
        return;
    }

    let f = frac_ponta.clamp(1.0 / (2.0 * p as f64), 0.45);
    let k1 = ((total as f64) * f) as usize;
    let k2 = total - k1;
    let nm = p - 2;

    let mut fr: Vec<usize> = Vec::with_capacity(p + 1);
    fr.push(0);
    for i in 0..=nm {
        fr.push(k1 + (k2 - k1) * i / nm);
    }
    fr.push(total);

    let cortes: Vec<(usize, usize)> = (1..p - 1)
        .into_par_iter()
        .map(|i| corte_diagonal(fr[i], a, b))
        .collect();

    let mut pedacos: Vec<&mut [T]> = Vec::with_capacity(p);
    let mut resto: &mut [T] = dest;
    for i in 0..p - 1 {
        let (esq, dir) = resto.split_at_mut(fr[i + 1] - fr[i]);
        pedacos.push(esq);
        resto = dir;
    }
    pedacos.push(resto);

    pedacos.into_par_iter().enumerate().for_each(|(i, d)| {
        if i == 0 {
            merge_front(a, b, d);
        } else if i == p - 1 {
            merge_back(a, b, d);
        } else {
            let (ia, ib) = cortes[i - 1];
            merge_front(&a[ia..], &b[ib..], d);
        }
    });
}

pub fn pway_merge_gallop<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    p: usize,
    frac_ponta: f64,
) {
    use crate::pim_kernel::{pim_back, pim_front};
    let total = a.len() + b.len();
    debug_assert_eq!(dest.len(), total);

    if p < 3 || total < p * 64 {
        let k = total / 2;
        let (df, db) = dest.split_at_mut(k);
        rayon::join(|| pim_front(a, b, df), || pim_back(a, b, db));
        return;
    }

    let f = frac_ponta.clamp(1.0 / (2.0 * p as f64), 0.45);
    let k1 = ((total as f64) * f) as usize;
    let k2 = total - k1;
    let nm = p - 2;

    let mut fr: Vec<usize> = Vec::with_capacity(p + 1);
    fr.push(0);
    for i in 0..=nm {
        fr.push(k1 + (k2 - k1) * i / nm);
    }
    fr.push(total);

    let cortes: Vec<(usize, usize)> = (1..p - 1)
        .into_par_iter()
        .map(|i| corte_diagonal(fr[i], a, b))
        .collect();

    let mut pedacos: Vec<&mut [T]> = Vec::with_capacity(p);
    let mut resto: &mut [T] = dest;
    for i in 0..p - 1 {
        let (esq, dir) = resto.split_at_mut(fr[i + 1] - fr[i]);
        pedacos.push(esq);
        resto = dir;
    }
    pedacos.push(resto);

    pedacos.into_par_iter().enumerate().for_each(|(i, d)| {
        if i == 0 {
            pim_front(a, b, d);
        } else if i == p - 1 {
            pim_back(a, b, d);
        } else {
            let (ia, ib) = cortes[i - 1];
            pim_front(&a[ia..], &b[ib..], d);
        }
    });
}

#[cfg(test)]
mod testes_pway {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct K { k: u32, i: u32 }
    impl PartialOrd for K { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
    impl Ord for K { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.k.cmp(&o.k) } }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn ate(&mut self, n: u64) -> u64 { if n == 0 { 0 } else { self.next() % n } }
    }

    #[test]
    fn pway_bate_a_referencia_em_varios_p_e_fracoes() {
        let mut r = Rng(0xBE5466CF34E90C6C);
        for t in 0..1500 {
            let (m, n) = (r.ate(900) as usize, r.ate(900) as usize);
            let card = [2u32, 11, 60, 99999][r.ate(4) as usize];
            let mut idx = 0u32;
            let mut a: Vec<K> = (0..m).map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x }).collect();
            let mut b: Vec<K> = (0..n).map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x }).collect();
            a.sort(); b.sort();

            let (mut ii, mut jj) = (0, 0);
            let mut esperado = Vec::with_capacity(m + n);
            while ii < m && jj < n {
                if a[ii] <= b[jj] { esperado.push(a[ii]); ii += 1 } else { esperado.push(b[jj]); jj += 1 }
            }
            esperado.extend_from_slice(&a[ii..]);
            esperado.extend_from_slice(&b[jj..]);
            let esp: Vec<_> = esperado.iter().map(|x| (x.k, x.i)).collect();

            let p = [3usize, 4, 8, 17, 32, 98][r.ate(6) as usize];
            let frac = [0.02f64, 0.05, 1.0 / p as f64, 0.2][r.ate(4) as usize];

            let mut g1 = vec![K { k: 0, i: 0 }; m + n];
            pway_merge(&a, &b, &mut g1, p, frac);
            assert_eq!(g1.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(), esp,
                       "pway_merge t={t} m={m} n={n} p={p} frac={frac}");

            let mut g2 = vec![K { k: 0, i: 0 }; m + n];
            pway_merge_gallop(&a, &b, &mut g2, p, frac);
            assert_eq!(g2.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(), esp,
                       "pway_merge_gallop t={t} m={m} n={n} p={p} frac={frac}");
        }
    }
}

pub const MIN_SEG: usize = 4096;

pub const TAREFAS_POR_THREAD: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plano {
    pub p: usize,
    pub k_frente: usize,
    pub k_tras: usize,
}

pub fn planeja(total: usize, threads: usize, r_tras: f64) -> Plano {
    let teto = (total / MIN_SEG).max(3);
    let p = (threads * TAREFAS_POR_THREAD).clamp(3, teto);
    let r = r_tras.clamp(0.25, 4.0);

    let s = (total as f64) / (p as f64 - 1.0 + 1.0 / r);
    let k_frente = (s.round() as usize).min(total);
    let k_tras = ((s / r).round() as usize).min(total - k_frente);
    Plano { p, k_frente, k_tras }
}

pub fn pway_merge_auto<T: Ord + Copy + Send + Sync>(
    a: &[T], b: &[T], dest: &mut [T], threads: usize, r_tras: f64, galope: bool,
) {
    let total = a.len() + b.len();
    let pl = planeja(total, threads, r_tras);
    pway_merge_plano(a, b, dest, pl, galope);
}

pub fn pway_merge_plano<T: Ord + Copy + Send + Sync>(
    a: &[T], b: &[T], dest: &mut [T], pl: Plano, galope: bool,
) {
    use crate::pim_kernel::{pim_back, pim_front};
    let total = a.len() + b.len();
    debug_assert_eq!(dest.len(), total);
    let p = pl.p;

    if p < 3 || total < p * 64 {
        let k = total / 2;
        let (df, db) = dest.split_at_mut(k);
        if galope { rayon::join(|| pim_front(a, b, df), || pim_back(a, b, db)); }
        else { rayon::join(|| merge_front(a, b, df), || merge_back(a, b, db)); }
        return;
    }

    let k1 = pl.k_frente.min(total);
    let k2 = total - pl.k_tras.min(total - k1);
    let nm = p - 2;

    let mut fr: Vec<usize> = Vec::with_capacity(p + 1);
    fr.push(0);
    for i in 0..=nm { fr.push(k1 + (k2 - k1) * i / nm); }
    fr.push(total);

    let mut pedacos: Vec<&mut [T]> = Vec::with_capacity(p);
    let mut resto: &mut [T] = dest;
    for i in 0..p - 1 {
        let (esq, dir) = resto.split_at_mut(fr[i + 1] - fr[i]);
        pedacos.push(esq);
        resto = dir;
    }
    pedacos.push(resto);

    pedacos.into_par_iter().enumerate().for_each(|(i, d)| {
        if i == 0 {
            if galope { pim_front(a, b, d) } else { merge_front(a, b, d) }
        } else if i == p - 1 {
            if galope { pim_back(a, b, d) } else { merge_back(a, b, d) }
        } else {
            let (ia, ib) = corte_diagonal(fr[i], a, b);
            if galope { pim_front(&a[ia..], &b[ib..], d) } else { merge_front(&a[ia..], &b[ib..], d) }
        }
    });
}

#[cfg(test)]
mod testes_auto {
    use super::*;

    #[test]
    fn plano_respeita_os_limites_e_o_modelo() {
        assert_eq!(planeja(1_000_000, 8, 1.0).p, 32);
        assert_eq!(planeja(100_000, 8, 1.0).p, 24);
        assert_eq!(planeja(1_000_000, 64, 1.0).p, 244);
        assert_eq!(planeja(5_000, 8, 1.0).p, 3);

        let pl = planeja(1_000_000, 8, 1.0);
        assert_eq!(pl.k_frente, pl.k_tras);

        let pl = planeja(1_000_000, 8, 1.5);
        let razao = pl.k_frente as f64 / pl.k_tras as f64;
        assert!((razao - 1.5).abs() < 0.02, "razao {razao}");

        let pl = planeja(1_000_000, 8, 1.5);
        let miolo = 1_000_000 - pl.k_frente - pl.k_tras;
        let s = miolo / (pl.p - 2);
        assert!((s as f64 / pl.k_frente as f64 - 1.0).abs() < 0.02, "miolo {s} frente {}", pl.k_frente);
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct K { k: u32, i: u32 }
    impl PartialOrd for K { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
    impl Ord for K { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.k.cmp(&o.k) } }
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn ate(&mut self, n: u64) -> u64 { if n == 0 { 0 } else { self.next() % n } }
    }

    #[test]
    fn pway_auto_bate_a_referencia() {
        let mut r = Rng(0x452821E638D01377);
        for t in 0..1200 {
            let (m, n) = (r.ate(1200) as usize, r.ate(1200) as usize);
            let card = [2u32, 11, 60, 99999][r.ate(4) as usize];
            let mut idx = 0u32;
            let mut a: Vec<K> = (0..m).map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x }).collect();
            let mut b: Vec<K> = (0..n).map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x }).collect();
            a.sort(); b.sort();
            let (mut ii, mut jj) = (0, 0);
            let mut esp = Vec::with_capacity(m + n);
            while ii < m && jj < n {
                if a[ii] <= b[jj] { esp.push(a[ii]); ii += 1 } else { esp.push(b[jj]); jj += 1 }
            }
            esp.extend_from_slice(&a[ii..]); esp.extend_from_slice(&b[jj..]);
            let e: Vec<_> = esp.iter().map(|x| (x.k, x.i)).collect();

            let threads = [1usize, 2, 8, 32][r.ate(4) as usize];
            let rr = [1.0f64, 1.5, 0.8][r.ate(3) as usize];
            for galope in [false, true] {
                let mut g = vec![K { k: 0, i: 0 }; m + n];
                pway_merge_auto(&a, &b, &mut g, threads, rr, galope);
                assert_eq!(g.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(), e,
                           "t={t} m={m} n={n} threads={threads} r={rr} galope={galope}");
            }
        }
    }
}

pub fn pway_merge_frente<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    p: usize,
    galope: bool,
) {
    use crate::pim_kernel::pim_front;
    let total = a.len() + b.len();
    debug_assert_eq!(dest.len(), total);

    if p < 2 || total < p * 64 {
        if galope { pim_front(a, b, dest) } else { merge_front(a, b, dest) }
        return;
    }

    let fr: Vec<usize> = (0..=p).map(|i| total * i / p).collect();

    let mut pedacos: Vec<&mut [T]> = Vec::with_capacity(p);
    let mut resto: &mut [T] = dest;
    for i in 0..p - 1 {
        let (esq, dir) = resto.split_at_mut(fr[i + 1] - fr[i]);
        pedacos.push(esq);
        resto = dir;
    }
    pedacos.push(resto);

    pedacos.into_par_iter().enumerate().for_each(|(i, d)| {
        if i == 0 {
            if galope { pim_front(a, b, d) } else { merge_front(a, b, d) }
        } else {
            let (ia, ib) = corte_diagonal(fr[i], a, b);
            if galope { pim_front(&a[ia..], &b[ib..], d) } else { merge_front(&a[ia..], &b[ib..], d) }
        }
    });
}

pub fn pway_merge_frente_auto<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    threads: usize,
    galope: bool,
) {
    let total = a.len() + b.len();
    let teto = (total / MIN_SEG).max(2);
    let p = (threads * TAREFAS_POR_THREAD).clamp(2, teto);
    pway_merge_frente(a, b, dest, p, galope);
}

pub fn pway_merge_frente_adaptativo<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    p: usize,
    min_gallop: usize,
) {
    use crate::pim_kernel::pim_front_adaptativo;

    let total = a.len() + b.len();
    debug_assert_eq!(dest.len(), total);
    if p < 2 || total < p * 64 {
        pim_front_adaptativo(a, b, dest, min_gallop);
        return;
    }

    let fr: Vec<usize> = (0..=p).map(|i| total * i / p).collect();
    let mut pedacos: Vec<&mut [T]> = Vec::with_capacity(p);
    let mut resto = dest;
    for i in 0..p - 1 {
        let (esq, dir) = resto.split_at_mut(fr[i + 1] - fr[i]);
        pedacos.push(esq);
        resto = dir;
    }
    pedacos.push(resto);

    pedacos.into_par_iter().enumerate().for_each(|(i, d)| {
        if i == 0 {
            pim_front_adaptativo(a, b, d, min_gallop);
        } else {
            let (ia, ib) = corte_diagonal(fr[i], a, b);
            pim_front_adaptativo(&a[ia..], &b[ib..], d, min_gallop);
        }
    });
}

pub fn pway_merge_frente_auto_adaptativo<T: Ord + Copy + Send + Sync>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    workers: usize,
    min_segmento: usize,
    tarefas_por_worker: usize,
    min_gallop: usize,
) {
    let total = a.len() + b.len();
    let teto = total / min_segmento.max(1);
    if workers < 2 || teto < 2 {
        crate::pim_kernel::pim_front_adaptativo(a, b, dest, min_gallop);
        return;
    }
    let p = workers
        .saturating_mul(tarefas_por_worker.max(1))
        .clamp(2, teto);
    pway_merge_frente_adaptativo(a, b, dest, p, min_gallop);
}

#[cfg(test)]
mod testes_frente {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct K { k: u32, i: u32 }
    impl PartialOrd for K { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
    impl Ord for K { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.k.cmp(&o.k) } }
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn ate(&mut self, n: u64) -> u64 { if n == 0 { 0 } else { self.next() % n } }
    }

    #[test]
    fn pway_frente_bate_a_referencia() {
        let mut r = Rng(0xC0AC29B7C97C50DD);
        for t in 0..1500 {
            let (m, n) = (r.ate(1200) as usize, r.ate(1200) as usize);
            let card = [2u32, 11, 60, 99999][r.ate(4) as usize];
            let mut idx = 0u32;
            let mut a: Vec<K> = (0..m).map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x }).collect();
            let mut b: Vec<K> = (0..n).map(|_| { let x = K { k: r.ate(card as u64) as u32, i: idx }; idx += 1; x }).collect();
            a.sort(); b.sort();
            let (mut ii, mut jj) = (0, 0);
            let mut esp = Vec::with_capacity(m + n);
            while ii < m && jj < n {
                if a[ii] <= b[jj] { esp.push(a[ii]); ii += 1 } else { esp.push(b[jj]); jj += 1 }
            }
            esp.extend_from_slice(&a[ii..]); esp.extend_from_slice(&b[jj..]);
            let e: Vec<_> = esp.iter().map(|x| (x.k, x.i)).collect();

            let p = [2usize, 3, 8, 17, 32, 98][r.ate(6) as usize];
            for galope in [false, true] {
                let mut g = vec![K { k: 0, i: 0 }; m + n];
                pway_merge_frente(&a, &b, &mut g, p, galope);
                assert_eq!(g.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(), e,
                           "frente t={t} m={m} n={n} p={p} galope={galope}");
                let mut g = vec![K { k: 0, i: 0 }; m + n];
                pway_merge_frente_auto(&a, &b, &mut g, [1, 2, 8, 32][r.ate(4) as usize], galope);
                assert_eq!(g.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>(), e, "frente_auto t={t}");
            }
        }
    }
}
