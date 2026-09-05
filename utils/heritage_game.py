"""
heritage_game.py

Heritage Matrix:
A Candy-Crush-style Match-3 puzzle themed around Indian culture.

Features:
- 6x6 Match-3 board
- Swap adjacent tiles
- Invalid swaps are reversed
- Matches of 3+ horizontally or vertically are cleared
- Tiles fall down using gravity
- New tiles appear from the top
- Automatic cascade matches
- Score is awarded for every clearing
- ONE new heritage fact is unlocked for EACH clearing event
- Starting board contains no matches
"""

import json


# ---------------------------------------------------------
# TILE SET
# ---------------------------------------------------------

TILE_EMOJIS = [
    "🪔",  # Diya
    "🎭",  # Theatre / performing arts
    "🥻",  # Saree
    "🛕",  # Temple
    "🎶",  # Music
    "🍛",  # Indian cuisine
    "🐘",  # Elephant
    "🌸",  # Lotus / flower
]


# ---------------------------------------------------------
# GAME RENDERER
# ---------------------------------------------------------

def render_heritage_matrix(facts, grid_size=6, height=640):

    facts_json = json.dumps(
        facts if facts else [
            "Explore Indian heritage by matching tiles!"
        ]
    )

    tiles_json = json.dumps(TILE_EMOJIS)

    return f"""
<div id="heritage-matrix-root">

<style>

#heritage-matrix-root {{
    font-family: Arial, sans-serif;
    text-align: center;
    max-width: 520px;
    margin: 0 auto;
    padding: 10px;
    box-sizing: border-box;
}}


/* ---------------------------------------------------------
   SCORE
--------------------------------------------------------- */

#hm-score {{
    font-size: 1.3rem;
    font-weight: 700;
    color: #B5651D;
    margin: 8px 0 12px;
}}


/* ---------------------------------------------------------
   BOARD
--------------------------------------------------------- */

#hm-board {{
    display: grid;
    grid-template-columns: repeat({grid_size}, 1fr);
    gap: 6px;

    width: min(420px, 92vw);

    margin: 0 auto;

    background: #FFF3E0;

    padding: 10px;

    border-radius: 16px;

    box-sizing: border-box;

    user-select: none;
}}


/* ---------------------------------------------------------
   TILE
--------------------------------------------------------- */

.hm-tile {{
    aspect-ratio: 1 / 1;

    display: flex;
    align-items: center;
    justify-content: center;

    background: white;

    border-radius: 10px;

    border: 2px solid transparent;

    font-size: clamp(1.5rem, 6vw, 2rem);

    cursor: pointer;

    transition:
        transform 0.12s ease,
        box-shadow 0.12s ease,
        opacity 0.15s ease;
}}


.hm-tile:hover {{
    transform: scale(1.06);
}}


.hm-tile.selected {{
    border-color: #FF9933;

    box-shadow:
        0 0 8px rgba(255, 153, 51, 0.8);

    transform: scale(1.08);
}}


.hm-tile.matched {{
    transform: scale(0.8);
    opacity: 0.25;
}}


/* ---------------------------------------------------------
   FACT BOX
--------------------------------------------------------- */

#hm-fact {{
    margin: 14px auto 0;

    padding: 12px 16px;

    background: #E8F5E9;

    border-left: 4px solid #138808;

    border-radius: 8px;

    min-height: 45px;

    max-width: 420px;

    box-sizing: border-box;

    font-size: 0.95rem;

    line-height: 1.35;

    color: #0B1F3A;
}}


/* ---------------------------------------------------------
   STATUS
--------------------------------------------------------- */

#hm-status {{
    margin-top: 8px;

    font-size: 0.82rem;

    color: #666;

    min-height: 18px;
}}


/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 450px) {{

    #hm-board {{
        gap: 4px;
        padding: 7px;
    }}

    .hm-tile {{
        border-radius: 8px;
    }}

}}

</style>


<!-- ------------------------------------------------------
     SCORE
------------------------------------------------------- -->

<div id="hm-score">

    Score:
    <span id="hm-score-val">0</span>

    &nbsp; | &nbsp;

    Facts unlocked:
    <span id="hm-facts-val">0</span>

</div>


<!-- ------------------------------------------------------
     BOARD
------------------------------------------------------- -->

<div id="hm-board"></div>


<!-- ------------------------------------------------------
     FACT
------------------------------------------------------- -->

<div id="hm-fact">

    Match 3 tiles in a row or column
    to unlock your first heritage fact!

</div>


<div id="hm-status"></div>


<script>


// =========================================================
// GAME DATA
// =========================================================

const SIZE = {grid_size};

const TILES = {tiles_json};

const FACTS = {facts_json};


// =========================================================
// GAME STATE
// =========================================================

let board = [];

let score = 0;

let factsUnlocked = 0;

let selected = null;

let busy = false;


// =========================================================
// RANDOM TILE
// =========================================================

function randomTile() {{

    return TILES[
        Math.floor(Math.random() * TILES.length)
    ];

}}


// =========================================================
// CHECK ADJACENCY
// =========================================================

function isAdjacent(a, b) {{

    const rowA = Math.floor(a / SIZE);
    const colA = a % SIZE;

    const rowB = Math.floor(b / SIZE);
    const colB = b % SIZE;

    const distance =
        Math.abs(rowA - rowB) +
        Math.abs(colA - colB);

    return distance === 1;

}}


// =========================================================
// SWAP
// =========================================================

function swap(a, b) {{

    const temp = board[a];

    board[a] = board[b];

    board[b] = temp;

}}


// =========================================================
// FIND ALL MATCHES
// =========================================================

function findMatches() {{

    const matches = new Set();


    // -----------------------------------------------------
    // HORIZONTAL MATCHES
    // -----------------------------------------------------

    for (let row = 0; row < SIZE; row++) {{

        let runStart = 0;

        for (
            let col = 1;
            col <= SIZE;
            col++
        ) {{

            const current =
                col < SIZE
                    ? board[row * SIZE + col]
                    : null;

            const previous =
                board[row * SIZE + (col - 1)];


            if (current !== previous) {{

                const runLength =
                    col - runStart;


                if (runLength >= 3) {{

                    for (
                        let x = runStart;
                        x < col;
                        x++
                    ) {{

                        matches.add(
                            row * SIZE + x
                        );

                    }}

                }}


                runStart = col;

            }}

        }}

    }}


    // -----------------------------------------------------
    // VERTICAL MATCHES
    // -----------------------------------------------------

    for (let col = 0; col < SIZE; col++) {{

        let runStart = 0;

        for (
            let row = 1;
            row <= SIZE;
            row++
        ) {{

            const current =
                row < SIZE
                    ? board[row * SIZE + col]
                    : null;

            const previous =
                board[(row - 1) * SIZE + col];


            if (current !== previous) {{

                const runLength =
                    row - runStart;


                if (runLength >= 3) {{

                    for (
                        let x = runStart;
                        x < row;
                        x++
                    ) {{

                        matches.add(
                            x * SIZE + col
                        );

                    }}

                }}


                runStart = row;

            }}

        }}

    }}


    return matches;

}}


// =========================================================
// RENDER BOARD
// =========================================================

function renderBoard(matchedIndexes = new Set()) {{

    const boardElement =
        document.getElementById("hm-board");


    boardElement.innerHTML = "";


    board.forEach((tile, index) => {{

        const tileElement =
            document.createElement("div");


        tileElement.className =
            "hm-tile";


        if (selected === index) {{

            tileElement.classList.add(
                "selected"
            );

        }}


        if (matchedIndexes.has(index)) {{

            tileElement.classList.add(
                "matched"
            );

        }}


        tileElement.innerText = tile;


        tileElement.onclick = function() {{

            handleTileClick(index);

        }};


        boardElement.appendChild(
            tileElement
        );

    }});

}}


// =========================================================
// UPDATE STATUS
// =========================================================

function setStatus(message) {{

    document.getElementById(
        "hm-status"
    ).innerText = message;

}}


// =========================================================
// UNLOCK EXACTLY ONE FACT
// =========================================================

function unlockOneFact() {{

    /*
       IMPORTANT:

       One clearing event = one fact.

       It does NOT matter whether the player
       cleared 3, 4, 5, or 10 tiles.

       Every time resolveMatches() finds a match,
       exactly ONE new fact is unlocked.
    */


    if (factsUnlocked >= FACTS.length) {{

        return;

    }}


    const fact =
        FACTS[factsUnlocked];


    document.getElementById(
        "hm-fact"
    ).innerText =
        "🔓 " + fact;


    factsUnlocked++;


    document.getElementById(
        "hm-facts-val"
    ).innerText =
        factsUnlocked;


    setStatus(
        "New heritage fact unlocked!"
    );

}}


// =========================================================
// APPLY GRAVITY
// =========================================================

function applyGravity() {{

    for (
        let col = 0;
        col < SIZE;
        col++
    ) {{

        const remaining = [];


        // -------------------------------------------------
        // Collect existing tiles from bottom upward
        // -------------------------------------------------

        for (
            let row = SIZE - 1;
            row >= 0;
            row--
        ) {{

            const value =
                board[row * SIZE + col];


            if (value !== null) {{

                remaining.push(value);

            }}

        }}


        // -------------------------------------------------
        // Generate new tiles
        // -------------------------------------------------

        while (
            remaining.length < SIZE
        ) {{

            remaining.push(
                randomTile()
            );

        }}


        // -------------------------------------------------
        // Put them back from bottom upward
        // -------------------------------------------------

        for (
            let row = SIZE - 1,
                i = 0;

            row >= 0;

            row--,
                i++
        ) {{

            board[row * SIZE + col] =
                remaining[i];

        }}

    }}

}}


// =========================================================
// RESOLVE MATCHES
// =========================================================

function resolveMatches() {{

    const matches =
        findMatches();


    // -----------------------------------------------------
    // NO MATCH
    // -----------------------------------------------------

    if (matches.size === 0) {{

        busy = false;

        selected = null;

        renderBoard();

        setStatus("");

        return;

    }}


    // -----------------------------------------------------
    // SCORE
    // -----------------------------------------------------

    const clearedCount =
        matches.size;


    score +=
        clearedCount * 10;


    document.getElementById(
        "hm-score-val"
    ).innerText =
        score;


    // -----------------------------------------------------
    // ONE FACT FOR THIS CLEARING
    // -----------------------------------------------------

    unlockOneFact();


    // -----------------------------------------------------
    // SHOW MATCHED TILES
    // -----------------------------------------------------

    renderBoard(matches);


    // -----------------------------------------------------
    // REMOVE MATCHED TILES
    // -----------------------------------------------------

    setTimeout(function() {{

        matches.forEach(
            function(index) {{

                board[index] = null;

            }}
        );


        renderBoard();


        // -------------------------------------------------
        // GRAVITY
        // -------------------------------------------------

        setTimeout(function() {{

            applyGravity();


            renderBoard();


            // ---------------------------------------------
            // CHECK FOR CASCADE
            // ---------------------------------------------

            setTimeout(function() {{

                resolveMatches();

            }}, 180);

        }}, 180);

    }}, 220);

}}


// =========================================================
// HANDLE PLAYER CLICK
// =========================================================

function handleTileClick(index) {{

    // Ignore clicks during animation
    if (busy) {{

        return;

    }}


    // -----------------------------------------------------
    // FIRST TILE
    // -----------------------------------------------------

    if (selected === null) {{

        selected = index;

        setStatus(
            "Choose an adjacent tile to swap."
        );

        renderBoard();

        return;

    }}


    // -----------------------------------------------------
    // CLICK SAME TILE
    // -----------------------------------------------------

    if (selected === index) {{

        selected = null;

        setStatus("");

        renderBoard();

        return;

    }}


    // -----------------------------------------------------
    // NON-ADJACENT TILE
    // -----------------------------------------------------

    if (!isAdjacent(selected, index)) {{

        selected = index;

        setStatus(
            "Choose an adjacent tile."
        );

        renderBoard();

        return;

    }}


    // -----------------------------------------------------
    // ATTEMPT SWAP
    // -----------------------------------------------------

    const first = selected;

    const second = index;


    swap(first, second);


    // Check whether swap creates a match
    const matches =
        findMatches();


    // -----------------------------------------------------
    // INVALID MOVE
    // -----------------------------------------------------

    if (matches.size === 0) {{

        // Undo swap
        swap(first, second);


        selected = null;


        setStatus(
            "That swap doesn't make a match."
        );


        renderBoard();


        return;

    }}


    // -----------------------------------------------------
    // VALID MOVE
    // -----------------------------------------------------

    selected = null;

    busy = true;


    setStatus(
        "Great match!"
    );


    renderBoard();


    // Start clearing
    setTimeout(function() {{

        resolveMatches();

    }}, 150);

}}


// =========================================================
// CREATE INITIAL BOARD
// =========================================================

function createInitialBoard() {{

    /*
       Keep generating until we get a board
       with NO matches already present.

       This prevents the game from starting
       with free automatic matches.
    */


    do {{

        board = [];


        for (
            let i = 0;
            i < SIZE * SIZE;
            i++
        ) {{

            board.push(
                randomTile()
            );

        }}

    }} while (
        findMatches().size > 0
    );

}}


// =========================================================
// START / RESET GAME
// =========================================================

function initBoard() {{

    score = 0;

    factsUnlocked = 0;

    selected = null;

    busy = false;


    createInitialBoard();


    document.getElementById(
        "hm-score-val"
    ).innerText = "0";


    document.getElementById(
        "hm-facts-val"
    ).innerText = "0";


    document.getElementById(
        "hm-fact"
    ).innerText =
        "Match 3 tiles in a row or column to unlock your first heritage fact!";


    setStatus("");


    renderBoard();

}}


// =========================================================
// START GAME
// =========================================================

initBoard();

</script>

</div>
"""