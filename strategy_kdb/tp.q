// Personal tick system

if[not system"p"; system"p 6010"]
system "l strategy_kdb/tick/",(src:first .z.x, enlist "sym"),".q"

system "l strategy_kdb/tick/u.q"
\d .u
dir: "strategy_kdb"
ld: {[date]
    if[not type key L::`$(-10_string L),string date; .[L;();:;()]];
    i::j::-11!(-2;L);
    hopen L};
tick: {
    init[];
    if[min[{in[`time`sym;raze x]}each cols each t]~0;
       `$"no timesym"; exit 1];
    @[;`sym;`g#] each t; d::.z.D;
    if[not count y; L:: `$":",y,dir,"/",x,10#"."]; l::ld[d]};

endofday: {
    end[d];
    d+:1;
    if[l; hclose l; l:: ld[d]]
 }
ts:{if[d<x;if[d<x-1;system"t 0";'"more than one day?"];endofday[]]};

if[not system "t";system "t 1000";
  .z.ts:{ts[.z.D]};
  upd: {[t; data]
      ts["d"$a:.z.P];
      if[not -16=type first first data; a:"n"$a; data:(enlist (count first data)#a), data];
    f: key flip value t;
    pub[t; $[0>type first data; enlist f!data; flip f!data]]
    if[.u.l;.u.l enlist (`upd;t;data);.u.i+:1];
   };];

\d .
.u.tick[src; .z.x 1]

