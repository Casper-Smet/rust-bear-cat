pub fn selection_sort(arr: &mut [i64]) -> &[i64] {
    for x in 0..arr.len() {
        let mut small = x;
        for y in (x + 1)..arr.len() {
            if arr[y] < arr[small] {
                // smaller than current smallest number
                small = y;
            }
        }
        arr.swap(small, x);
    }
    arr
}
