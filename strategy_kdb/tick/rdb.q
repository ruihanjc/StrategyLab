if[not system "p"; system "p 5011"]

upd:insert

.u.x: ("::5010"; "::5012") / TP and HDB ports

.u.end: {[date]
    t: tables`.;
    t@: where `g=attr each {x[`sym]} each t
    .Q.hdpf[`$"::5012"; `:.; date; `sym];
    @[;`sym;`g#] each t;};

.u.rep:{[x;y](.[;();:;].)each x; system "cd ",1_-10_string last y};


.u.rep . (hopen first .u.x)"(.u.sub[`;`];`.u `i`L)"


selectFunc:{[tbl;st;et;syms]
     select from tbl where date within st et, sym in syms
 }