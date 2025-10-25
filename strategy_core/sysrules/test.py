from collections import defaultdict


def get_VWAP():

    n = int(input())

    ticks_by_symbol = defaultdict(list)
    latest_timestamp = 0

    for _ in range(n):
        line = input().split()
        timestamp = int(line[0])
        symbol = line[1]
        price = float(line[2])
        vol = int(line[3])

        ticks_by_symbol[symbol].append((timestamp, price, vol))

        latest_timestamp = max(latest_timestamp, timestamp)


    m = int(input())

    for _ in range(m):
        line = input().split()
        symbol = line[0]
        seconds = int(line[1])


        total_value = 0
        total_volume = 0
        symbol_queries = ticks_by_symbol[symbol]
        symbol_queries.sort(key = lambda x : x[0], reverse = True)
        minimum_time = latest_timestamp - seconds

        for entry in symbol_queries:
            if entry[0] > minimum_time:
                total_value += entry[1]
                total_volume += entry[2]

        print(f"{(total_value + total_volume) / total_volume:.2f}")