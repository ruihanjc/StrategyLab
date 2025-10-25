if[not system "p"; system "p 5011"]

.u.end: {[date]
    t: tables`.;
    t@: where `g=attr each {x[`sym]} each t
    .Q.hdpf[`$"::5012"; `$.u.dir; date; `sym];
    @[;`sym;`g#] each t;};