#[inline(always)]
fn le<T: Ord>(a: &T, b: &T) -> bool { a <= b }

#[inline]
pub fn merge_ro<T: Ord + Copy>(a: &[T], b: &[T], dest: &mut [T]) {
    let (mut i, mut j) = (0usize, 0usize);
    for slot in dest.iter_mut() {
        let take_a = j >= b.len() || (i < a.len() && le(&a[i], &b[j]));
        if take_a { *slot = a[i]; i += 1; } else { *slot = b[j]; j += 1; }
    }
}

#[inline]
fn gallop_upper<T: Ord + Copy>(s: &[T], key: &T) -> usize {
    let n = s.len();
    if n == 0 { return 0; }
    if s[0] > *key { return 0; }
    let mut hi = 1usize;
    while hi < n && s[hi] <= *key { hi <<= 1; }
    let lo = hi >> 1;
    let hi = hi.min(n);
    lo + s[lo..hi].partition_point(|x| x <= key)
}

#[inline]
fn gallop_lower<T: Ord + Copy>(s: &[T], key: &T) -> usize {
    let n = s.len();
    if n == 0 { return 0; }
    if s[0] >= *key { return 0; }
    let mut hi = 1usize;
    while hi < n && s[hi] < *key { hi <<= 1; }
    let lo = hi >> 1;
    let hi = hi.min(n);
    lo + s[lo..hi].partition_point(|x| x < key)
}

#[inline]
fn gallop_lower_rev<T: Ord + Copy>(s: &[T], key: &T) -> usize {
    let n = s.len();
    if n == 0 { return 0; }
    if s[n - 1] < *key { return n; }
    let mut step = 1usize;
    while step < n && s[n - 1 - step] >= *key { step <<= 1; }
    let lo = n - step.min(n);
    let hi = n - (step >> 1);
    lo + s[lo..hi].partition_point(|x| x < key)
}

#[inline]
fn gallop_upper_rev<T: Ord + Copy>(s: &[T], key: &T) -> usize {
    let n = s.len();
    if n == 0 { return 0; }
    if s[n - 1] <= *key { return n; }
    let mut step = 1usize;
    while step < n && s[n - 1 - step] > *key { step <<= 1; }
    let lo = n - step.min(n);
    let hi = n - (step >> 1);
    lo + s[lo..hi].partition_point(|x| x <= key)
}

pub fn pim_front<T: Ord + Copy>(a: &[T], b: &[T], dest: &mut [T]) {
    let (mut ia, mut ib, mut k) = (0usize, 0usize, 0usize);
    let want = dest.len();
    while k < want {
        if ia == a.len() { let n = want - k; dest[k..].copy_from_slice(&b[ib..ib + n]); return; }
        if ib == b.len() { let n = want - k; dest[k..].copy_from_slice(&a[ia..ia + n]); return; }
        let na = gallop_upper(&a[ia..], &b[ib]).min(want - k);
        if na > 0 {
            dest[k..k + na].copy_from_slice(&a[ia..ia + na]);
            k += na; ia += na;
            if k == want || ia == a.len() { continue; }
        }
        let nb = gallop_lower(&b[ib..], &a[ia]).min(want - k);
        dest[k..k + nb].copy_from_slice(&b[ib..ib + nb]);
        k += nb; ib += nb;
    }
}

pub fn pim_back<T: Ord + Copy>(a: &[T], b: &[T], dest: &mut [T]) {
    let (mut qa, mut qb) = (a.len(), b.len());
    let mut k = dest.len();
    while k > 0 {
        if qa == 0 { dest[..k].copy_from_slice(&b[qb - k..qb]); return; }
        if qb == 0 { dest[..k].copy_from_slice(&a[qa - k..qa]); return; }
        let j = gallop_lower_rev(&b[..qb], &a[qa - 1]);
        let nb = (qb - j).min(k);
        if nb > 0 {
            dest[k - nb..k].copy_from_slice(&b[qb - nb..qb]);
            k -= nb; qb -= nb;
            if k == 0 || qb == 0 { continue; }
        }
        let i = gallop_upper_rev(&a[..qa], &b[qb - 1]);
        let na = (qa - i).min(k);
        dest[k - na..k].copy_from_slice(&a[qa - na..qa]);
        k -= na; qa -= na;
    }
}

pub fn pim_front_adaptativo<T: Ord + Copy>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    min_gallop: usize,
) {
    let (mut ia, mut ib, mut k) = (0usize, 0usize, 0usize);
    let limite = min_gallop.max(1);
    let mut vitorias_a = 0usize;
    let mut vitorias_b = 0usize;

    while k < dest.len() {
        if ia == a.len() {
            let restantes = dest.len() - k;
            dest[k..].copy_from_slice(&b[ib..ib + restantes]);
            return;
        }
        if ib == b.len() {
            let restantes = dest.len() - k;
            dest[k..].copy_from_slice(&a[ia..ia + restantes]);
            return;
        }

        if a[ia] <= b[ib] {
            dest[k] = a[ia];
            ia += 1;
            k += 1;
            vitorias_a += 1;
            vitorias_b = 0;

            if vitorias_a >= limite && ia < a.len() && ib < b.len() && k < dest.len() {
                let n = gallop_upper(&a[ia..], &b[ib]).min(dest.len() - k);
                if n > 0 {
                    dest[k..k + n].copy_from_slice(&a[ia..ia + n]);
                    ia += n;
                    k += n;
                }
                vitorias_a = 0;
            }
        } else {
            dest[k] = b[ib];
            ib += 1;
            k += 1;
            vitorias_b += 1;
            vitorias_a = 0;

            if vitorias_b >= limite && ia < a.len() && ib < b.len() && k < dest.len() {
                let n = gallop_lower(&b[ib..], &a[ia]).min(dest.len() - k);
                if n > 0 {
                    dest[k..k + n].copy_from_slice(&b[ib..ib + n]);
                    ib += n;
                    k += n;
                }
                vitorias_b = 0;
            }
        }
    }
}

pub fn pim_back_adaptativo<T: Ord + Copy>(
    a: &[T],
    b: &[T],
    dest: &mut [T],
    min_gallop: usize,
) {
    let (mut qa, mut qb, mut k) = (a.len(), b.len(), dest.len());
    let limite = min_gallop.max(1);
    let mut vitorias_a = 0usize;
    let mut vitorias_b = 0usize;

    while k > 0 {
        if qa == 0 {
            dest[..k].copy_from_slice(&b[qb - k..qb]);
            return;
        }
        if qb == 0 {
            dest[..k].copy_from_slice(&a[qa - k..qa]);
            return;
        }

        if b[qb - 1] >= a[qa - 1] {
            qb -= 1;
            k -= 1;
            dest[k] = b[qb];
            vitorias_b += 1;
            vitorias_a = 0;

            if vitorias_b >= limite && qa > 0 && qb > 0 && k > 0 {
                let inicio = gallop_lower_rev(&b[..qb], &a[qa - 1]);
                let n = (qb - inicio).min(k);
                if n > 0 {
                    dest[k - n..k].copy_from_slice(&b[qb - n..qb]);
                    qb -= n;
                    k -= n;
                }
                vitorias_b = 0;
            }
        } else {
            qa -= 1;
            k -= 1;
            dest[k] = a[qa];
            vitorias_a += 1;
            vitorias_b = 0;

            if vitorias_a >= limite && qa > 0 && qb > 0 && k > 0 {
                let inicio = gallop_upper_rev(&a[..qa], &b[qb - 1]);
                let n = (qa - inicio).min(k);
                if n > 0 {
                    dest[k - n..k].copy_from_slice(&a[qa - n..qa]);
                    qa -= n;
                    k -= n;
                }
                vitorias_a = 0;
            }
        }
    }
}

#[cfg(test)]
mod testes_adaptativo {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    struct K {
        chave: u32,
        ordem: u32,
    }
    impl PartialEq for K { fn eq(&self, outro: &Self) -> bool { self.chave == outro.chave } }
    impl Eq for K {}
    impl PartialOrd for K {
        fn partial_cmp(&self, outro: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(outro)) }
    }
    impl Ord for K { fn cmp(&self, outro: &Self) -> std::cmp::Ordering { self.chave.cmp(&outro.chave) } }

    #[test]
    fn frente_e_tras_adaptativos_preservam_estabilidade() {
        let a = [
            K { chave: 1, ordem: 0 }, K { chave: 2, ordem: 1 }, K { chave: 2, ordem: 2 },
            K { chave: 7, ordem: 3 }, K { chave: 9, ordem: 4 },
        ];
        let b = [
            K { chave: 2, ordem: 5 }, K { chave: 3, ordem: 6 }, K { chave: 6, ordem: 7 },
            K { chave: 7, ordem: 8 }, K { chave: 10, ordem: 9 },
        ];
        let mut esperado = [a[0]; 10];
        esperado[..5].copy_from_slice(&a);
        esperado[5..].copy_from_slice(&b);
        esperado.sort();

        let mut frente = [a[0]; 10];
        pim_front_adaptativo(&a, &b, &mut frente, 2);
        assert_eq!(
            frente.iter().map(|x| (x.chave, x.ordem)).collect::<Vec<_>>(),
            esperado.iter().map(|x| (x.chave, x.ordem)).collect::<Vec<_>>(),
        );

        let mut bidirecional = [a[0]; 10];
        let (df, dt) = bidirecional.split_at_mut(5);
        rayon::join(
            || pim_front_adaptativo(&a, &b, df, 2),
            || pim_back_adaptativo(&a, &b, dt, 2),
        );
        assert_eq!(
            bidirecional.iter().map(|x| (x.chave, x.ordem)).collect::<Vec<_>>(),
            esperado.iter().map(|x| (x.chave, x.ordem)).collect::<Vec<_>>(),
        );
    }
}
