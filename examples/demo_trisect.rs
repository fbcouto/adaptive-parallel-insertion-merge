use adaptive_parallel_insertion_merge::cut::{corte_diagonal, merge_back, merge_front, trisect_merge};

fn main() {
    let a: Vec<u32> = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
    let b: Vec<u32> = vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
    let total = a.len() + b.len();

    println!("A = {a:?}");
    println!("B = {b:?}");
    println!("total = {total}, dividido em 5 partes de {} -> 2/5 | 1/5 | 2/5\n", total / 5);

    let k1 = (total as f64 * 0.4) as usize;
    let k2 = total - k1;
    println!("thread 1 (FRENTE) emite dest[0..{k1}]   = {k1} elementos");
    println!("thread 2 (MEIO)   emite dest[{k1}..{k2}]  = {} elementos", k2 - k1);
    println!("thread 3 (TRAS)   emite dest[{k2}..{total}] = {} elementos\n", total - k2);

    println!("As duas PONTAS nao precisam de corte: o desempate complementar");
    println!("(frente -> A, tras -> B) garante que elas particionam o multiconjunto.");
    println!("So a thread do MEIO precisa saber onde comecar.\n");

    println!("corte_diagonal(k={k1}):  busca binaria em i, com j = {k1} - i");
    let (m, n) = (a.len(), b.len());
    let (mut lo, mut hi) = (k1.saturating_sub(n), k1.min(m));
    let mut passo = 0;
    while lo < hi {
        let i = lo + (hi - lo + 1) / 2;
        let j = k1 - i;
        let ok = j == n || a[i - 1] <= b[j];
        passo += 1;
        println!("  passo {passo}: i={i} j={j}  ->  a[{}]={} <= b[{}]={} ? {}",
                 i - 1, a[i - 1], j, if j < n { b[j] as i64 } else { -1 },
                 if ok { "SIM, i pode crescer" } else { "NAO, i deve encolher" });
        if ok { lo = i } else { hi = i - 1 }
    }
    let (i, j) = (lo, k1 - lo);
    assert_eq!((i, j), corte_diagonal(k1, &a, &b));
    println!("  resultado: (i, j) = ({i}, {j})   em {passo} comparacoes\n");

    println!("A[..{i}] = {:?}", &a[..i]);
    println!("B[..{j}] = {:?}   -> juntos, exatamente os {k1} menores", &b[..j]);
    println!("o {}o elemento e min(a[{i}], b[{j}]) = min({}, {}) = {}\n",
             k1 + 1, a[i], b[j], a[i].min(b[j]));

    let mut d1 = vec![0u32; k1];
    let mut d2 = vec![0u32; k2 - k1];
    let mut d3 = vec![0u32; total - k2];
    merge_front(&a, &b, &mut d1);
    merge_front(&a[i..], &b[j..], &mut d2);
    merge_back(&a, &b, &mut d3);
    println!("thread 1 (FRENTE, sem corte)      -> {d1:?}");
    println!("thread 2 (MEIO, a[{i}..] e b[{j}..])  -> {d2:?}");
    println!("thread 3 (TRAS, sem corte)        -> {d3:?}\n");

    let mut dest = vec![0u32; total];
    trisect_merge(&a, &b, &mut dest, 0.4);
    println!("trisect_merge(.., 0.4)            -> {dest:?}");
    assert!(dest.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(dest, (1..=20u32).collect::<Vec<_>>());
    println!("\nconfere: ordenado e igual a 1..20");
}
