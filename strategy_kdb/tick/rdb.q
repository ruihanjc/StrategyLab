if[not system "p"; system "p 5011"]

upd:insert

.u.x: ("::5010"; "::5012") / TP and HDB ports

.u.end: {[date]
    t: tables`.;
    t@: where `g=attr each {x[`sym]} each t
    .Q.hdpf[`$"::5012"; `:.; date; `sym];
    @[;`sym;`g#] each t;};

.u.rep:{[x;y](.[;();:;].)each x; (`upd; x; );system "cd ",1_-10_string last y};


.u.rep . (hopen first .u.x)"(.u.sub[`;`];`.u `i`L)"
/ .u.sub returns all the tables.
/ the second call is a nested list of value of .u.i and .u.L

selectFunc:{[tbl;st;et;syms]
     $[`date in cols tbl;
      select from tbl where date within (st;et),sym in syms;
      [res:$[.z.D within (st;et); select from tbl where sym in syms;0#value tbl];
        `date xcols update date:.z.D from res]] }