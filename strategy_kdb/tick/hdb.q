if[not system "p"; system "p 5012"]
if[1>count .z.x;show"Supply directory of historical database";exit 0];
hdb: .z.x 0
dir: "strategy_kdb/"
@[{system"l ",x};dir,hdb;{show "Error message - ",x;exit 0}]


selectFunc:{[tbl;st;et;syms]
     $[syms~`;
       select from tbl where date within (st;et);
       select from tbl where date within (st;et), sym in syms]
 }
