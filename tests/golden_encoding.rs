//! The encoder must reproduce the golden vectors (`vectors.npz`, shipped with each network)
//! bit for bit: the embedded default network's, fetched at build time, or the file
//! `MARINA_VECTORS` names. Skipped only when neither is available (a build without the
//! `embedded` feature and no override).

use std::io::Cursor;

use marina::encoding::{ACTION_SIZE, LegalActions, encode_board};
use marina::position::Game;
use shakmaty::CastlingMode;

fn read<T: npyz::Deserialize>(
    archive: &mut npyz::npz::NpzArchive<Cursor<Vec<u8>>>,
    name: &str,
) -> (Vec<u64>, Vec<T>) {
    let npy = archive
        .by_name(name)
        .expect("read array")
        .unwrap_or_else(|| panic!("missing array {name}"));
    let shape = npy.shape().to_vec();
    let data = npy.into_vec::<T>().expect("decode array");
    (shape, data)
}

#[test]
fn encoder_matches_golden_vectors() {
    let path = match std::env::var("MARINA_VECTORS") {
        Ok(path) => path,
        Err(_) => match marina::net::embedded::default() {
            Ok(net) => net.vectors.to_string(),
            Err(_) => {
                eprintln!("no embedded network and MARINA_VECTORS not set; skipping");
                return;
            }
        },
    };
    let bytes = std::fs::read(&path).expect("read vectors.npz");
    let mut archive = npyz::npz::NpzArchive::new(Cursor::new(bytes)).expect("open npz");

    let (_, fens) = read::<String>(&mut archive, "fen");
    let (_, pieces) = read::<u8>(&mut archive, "pieces");
    let (_, castling) = read::<u8>(&mut archive, "castling");
    let (_, en_passant) = read::<u8>(&mut archive, "en_passant");
    let (mask_shape, legal_mask) = read::<bool>(&mut archive, "legal_mask");
    let count = fens.len();
    assert_eq!(mask_shape, vec![count as u64, ACTION_SIZE as u64]);

    let mut failures = Vec::new();
    for (index, fen) in fens.iter().enumerate() {
        let game = Game::from_uci(Some(fen), &[], CastlingMode::Standard)
            .unwrap_or_else(|error| panic!("position {index} {fen}: {error}"));
        let encoded = encode_board(&game);
        if encoded.pieces[..] != pieces[index * 64..(index + 1) * 64] {
            failures.push(format!("{index} {fen}: pieces differ"));
        }
        if encoded.castling != castling[index] {
            failures.push(format!(
                "{index} {fen}: castling {} != {}",
                encoded.castling, castling[index]
            ));
        }
        if encoded.en_passant != en_passant[index] {
            failures.push(format!(
                "{index} {fen}: en_passant {} != {}",
                encoded.en_passant, en_passant[index]
            ));
        }
        let mask = LegalActions::of(&game).mask();
        let expected = &legal_mask[index * ACTION_SIZE..(index + 1) * ACTION_SIZE];
        if mask[..] != expected[..] {
            let ours: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter(|&(_, &b)| b)
                .map(|(i, _)| i)
                .collect();
            let theirs: Vec<usize> = expected
                .iter()
                .enumerate()
                .filter(|&(_, &b)| b)
                .map(|(i, _)| i)
                .collect();
            failures.push(format!(
                "{index} {fen}: legal mask differs\n  ours   {ours:?}\n  theirs {theirs:?}"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {count} positions differ:\n{}",
        failures.len(),
        failures
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    eprintln!("encoder matches {count} golden positions");
}
