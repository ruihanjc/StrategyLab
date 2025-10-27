/ Initialize with q gw.q userpsw -p 5050

if[not system "p"; system "p 5050"]

dir: "strategy_kdb/tick/"
.perm.users: ("s*s"; enlist csv) 0: hsym `$dir,(first .z.x),".csv";
.perm.accessLog: ([] username:0#`; handle:0#enlist "" ;timestamp: 0#.z.Z; open:0#0b);
.perm.executionLog: ([] username:0#`; handle:0#enlist "";timestamp: 0#.z.Z; execution: 0#enlist "";sync:0#0b);
sha1fy: {.Q.sha1 each x};
@[`.perm.users; `password; sha1fy];
`username xkey `.perm.users;

.z.pw: {[usr;psw] (.Q.sha1 psw)~(.perm.users[usr][`password]) }
.z.po: {[handle] `.perm.accessLog upsert (.z.u; string handle;.z.Z;1b) }
.z.pc: {[handle] `.perm.accessLog upsert (.z.u; string handle;.z.Z;0b) }
.z.pg: {[msg] `.perm.executionLog upsert (.z.u; .z.W ;.z.Z; string msg;1b); value msg}
.z.ps: {[msg] `.perm.executionLog upsert (.z.u; .z.W ;.z.Z; string msg;0b); value msg}


h_hdb: hopen `::5012
h_rdb: hopen `::5011

getTradeData:{[sd;ed;ids]
  @[h_hdb; (`selectFunc;`tickerData;sd;ed;ids); `$"No hdb table error"],
  @[h_rdb; (`selectFunc;`tickerData;sd;ed;ids); `$"No rdb table error"]}