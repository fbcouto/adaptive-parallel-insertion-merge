use adaptive_parallel_insertion_merge as api;

#[derive(Clone, Copy, Debug)]
struct K { k: u32, i: u32 }
impl PartialEq for K { fn eq(&self, o: &Self) -> bool { self.k == o.k } }
impl Eq for K {}
impl PartialOrd for K { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for K { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.k.cmp(&o.k) } }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn ate(&mut self, n: u64) -> u64 { if n == 0 { 0 } else { self.next() % n } }
}

fn main() {
    let mut r = Rng(0x2545F4914F6CDD1D);
    let mut falhas: Vec<(String, usize)> = vec![];

    for t in 0..300 {
        let n = 3_000 + r.ate(60_000) as usize;
        let card = [2u32, 25, 3000, 5_000_000][t % 4];
        let forma = t % 5;

        let mut base: Vec<K> = (0..n as u32).map(|i| K { k: r.ate(card as u64) as u32, i }).collect();
        match forma {
            1 => base.sort(),
            2 => { base.sort(); base.reverse(); }
            3 => { base.sort(); for _ in 0..n/200 { let (a,b)=(r.ate(n as u64) as usize, r.ate(n as u64) as usize); base.swap(a,b); } }
            4 => { let c = n/2; base[..c].sort(); }
            _ => {}
        }
        let mut esperado = base.clone();
        esperado.sort();
        let e: Vec<_> = esperado.iter().map(|x| (x.k, x.i)).collect();

        let bs = [64usize, 512, 1920, 8192][r.ate(4) as usize];
        let ss = [8usize, 32, 128][r.ate(3) as usize];
        let cost = r.ate(2) == 0;

        let mut checa = |nome: &str, v: &[K]| {
            if v.iter().map(|x| (x.k, x.i)).collect::<Vec<_>>() != e {
                let ordenado = v.windows(2).all(|w| w[0].k <= w[1].k);
                falhas.push((format!("{nome} (ordenado={ordenado}) forma={forma} n={n} card={card} bs={bs} cost={cost}"), t));
            }
        };

        let mut v = base.clone(); api::pim_sort(&mut v);                                  checa("pim_sort", &v);
        let mut v = base.clone(); api::pim_sort_pway(&mut v);                             checa("pim_sort_pway", &v);
        let mut v = base.clone(); api::pim_sort_pway_sem_escudo(&mut v);                  checa("pim_sort_pway_sem_escudo", &v);
        let mut v = base.clone(); api::pim_sort_pway_blocos(&mut v, bs, cost);            checa("pway_blocos", &v);
        let mut v = base.clone(); api::pim_sort_pway_blocos_sort_local(&mut v, bs, cost); checa("pway_blocos_sort_local", &v);
        let mut v = base.clone(); api::pim_sort_pway_blocos_subsorts(&mut v, bs, ss, cost); checa("pway_blocos_subsorts", &v);
        let mut v = base.clone(); api::pim_sort_pway_blocos_sem_costura(&mut v, bs);      checa("pway_blocos_sem_costura", &v);
    }

    if falhas.is_empty() {
        println!("OK — 300 trials x 7 pontos de entrada: ordem e ESTABILIDADE corretas.");
    } else {
        println!("{} FALHAS:", falhas.len());
        let mut vistos = std::collections::HashSet::new();
        for (f, t) in &falhas {
            let chave = f.split(" (").next().unwrap().to_string();
            if vistos.insert(chave) { println!("  trial {t}: {f}"); }
        }
    }
}
