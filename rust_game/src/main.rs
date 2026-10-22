

mod game;


#[allow(arithmetic_overflow)]
fn main() {
    println!("Hello, world!");
}


// use std::str::Bytes;
// fn countIslands(grid: Vec<String>) -> i32 {
//     let grid: Vec<Bytes> = grid.iter().map(|x| x.as_bytes()).collect();
//     for row in grid
// }
//


fn isPalindrome(text: String) -> bool {
    let text = text.to_lowercase().replace(" ", "");
    let text_bytes = text.as_bytes();
    let total_len = text_bytes.len();
    for i in 0..text_bytes.len() {
        if text_bytes[i] != text_bytes[total_len - 1 - i] {
            return false
        }
    }
    true
}