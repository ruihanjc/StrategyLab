

\d .u

init: {w::t!(count t::tables`.)#()} /Initialize a dictionary of tables mapped to each subscriber with it's subscribed columns
del: {[x;y] w[x]_: w[x;;0]?y} /Delete a subscribed entry in case of repetition
sel: {[x;y] $[`~y;x;select from x where sym in y]} /Select certain entries of the selected table
pub: {[t; tbl] {[t;tbl;pair] if[count tbl:sel[tbl] pair[1]; (neg pair[0]) (`upd; t; tbl)]}[t;tbl] each w[t]} /Publish the data in to the subscribers
add: {[x;y]  w[x],: enlist (.z.w; (y)); (x; @[0#value x; `sym;`g#])} / x: name of the table , y: List of symbols subscribed to
sub: {[x;y] if[x~`;:sub[;y] each t]; if[not x in t; `x]; del[x].z.w; add[x;y]} /Subscribe the handle with it's desired tables.
end: {[date] (neg distinct raze w)@\: (`u.end; date)}  /Send .u.end function to each of its subscribers as it's the end of the day.