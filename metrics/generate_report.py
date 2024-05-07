import csv
import requests
import sys
import math
from statistics import mean
import os

#TODO add volume to docker run to store events file and txt with results
EVENTS_FILE_PATH = os.getenv("EVENTS_FILE")
OUTPUT_FILE_PATH = os.getenv("REPORTS_OUTPUTS")

PROMETHEUS_URL = "http://localhost:9090"
QUERY_API = "/api/v1/query"
RANGE_QUERY_API = '/api/v1/query_range'
STEP = "1s"
RATE_INTERVAL = "1m"
GAUGE_METRICS = ["container_memory_usage_bytes", "container_memory_max_usage_bytes", "container_cpu_load_average_10s",
                 "container_fs_io_current"]
COUNTER_METRICS = ["container_fs_io_time_seconds_total", "container_memory_failcnt",
                   "container_cpu_usage_seconds_total"]

GAUGE_METRICS = ["container_memory_usage_bytes"]
COUNTER_METRICS = ["container_cpu_usage_seconds_total"]

CADVISOR_CONTAINER = "cadvisor"


def query_metric_names():
    response = requests.get(PROMETHEUS_URL + QUERY_API,
                            params={'query': 'sum by(__name__)({{name="{0}"}})'.format(CADVISOR_CONTAINER)})
    status = response.json()['status']

    if status == "error":
        print(response.json())
        sys.exit(2)

    results = response.json()['data']['result']
    metricnames = list()
    for result in results:
        metricnames.append(result['metric'].get('__name__', ''))
    metricnames.sort()

    return metricnames


GAUGE_SINGLE_CONTAINER_QUERY = '{metric}{{name="{container}"}}'
COUNTER_SINGLE_CONTAINER_QUERY = 'rate({metric}{{name="{container}"}}[{rate_interval}])'

GAUGE_MUTLIPLE_CONTAINERS_QUERY = 'sum({metric}{{name=~"{container}"}})'
COUNTER_MUTLIPLE_CONTAINERS_QUERY = 'sum(rate({metric}{{name=~"{container}"}}[{rate_interval}]))'


def query_metric_values(metric_names, start_time, end_time, container):
    values = {}

    for metric in metric_names:
        if metric in COUNTER_METRICS:
            query_str = COUNTER_SINGLE_CONTAINER_QUERY.format(metric=metric, container=container, rate_interval=RATE_INTERVAL)
            response = requests.get(PROMETHEUS_URL + RANGE_QUERY_API,
                                    params={'query': query_str,
                                            'start': start_time,
                                            'end': end_time,
                                            'step': STEP})
        else:
            query_str = GAUGE_SINGLE_CONTAINER_QUERY.format(metric=metric, container=container)
            response = requests.get(PROMETHEUS_URL + RANGE_QUERY_API,
                                    params={'query': query_str,
                                            'start': start_time,
                                            'end': end_time, 'step': STEP})
        print(f'Query: {query_str}')
        status = response.json()['status']

        print(f'request: {response.request.url}')

        if status == "error":
            print(response.json())
            sys.exit(2)

        results = response.json()['data']['result']

        # print(results)
        if len(results) == 0:
            print(response.json())
            sys.exit(2)

        values[metric] = results[0]['values']

    return values


def postprocess_and_upload(values, events, framework):
    metrics_per_label = groupLabels(events, values)

    # print(metrics_per_label)

    stats_per_action = [generate_stats(action[0], action[1]) for action in metrics_per_label]

    # print(stats_per_action)

    export_stats_to_csv(stats_per_action, framework)


def generate_ad_hoc_e2e_metrics(events, values):
    range_metrics = {}
    for metric in values:
        range_metrics[metric] = values[metric]

    e2e_range = (events[0][1], events[-1][1])
    return (("E2E", e2e_range), range_metrics)


def groupLabels(events, values):
    metrics_per_label = []

    for i in range(len(events) - 1):
        active_range = (int(events[i][1]), int(events[i + 1][1]))
        active_label = events[i + 1][0]

        range_metrics = {}

        for metric in values:
            matching_values = []

            for existing_value in values[metric]:
                if active_range[0] < existing_value[0] < active_range[1]:
                    matching_values.append(existing_value)

            range_metrics[metric] = matching_values
            print(
                f'Label: {active_label} - Range: {active_range}. Found: {len(matching_values)} matching values for {metric}')

        active_range_in_milis = (events[i][1], events[i + 1][1])

        metrics_per_label.append(((active_label, active_range_in_milis), range_metrics))

    metrics_per_label.append(generate_ad_hoc_e2e_metrics(events, values))

    return metrics_per_label


def generate_stats(action, values):
    stats = {}
    # print(f'{action} : {len(values)}')

    for metric_name in values:
        metrics = values[metric_name]
        metric_stats = {}

        # add stats
        metric_stats["AVG"] = mean(float(metric[1]) for metric in metrics) if len(metrics) != 0 else 0
        metric_stats["MAX"] = max(float(metric[1]) for metric in metrics) if len(metrics) != 0 else 0
        metric_stats["MIN"] = min(float(metric[1]) for metric in metrics) if len(metrics) != 0 else 0

        stats[metric_name] = metric_stats

    return action[0], action[1][1] - action[1][0], stats


def export_stats_to_csv(stats_per_action, framework):
    need_headers = not os.path.exists(OUTPUT_FILE_PATH)

    with open(OUTPUT_FILE_PATH, 'a', newline='') as file:
        writer = csv.writer(file)

        if need_headers:
            # Base headers + dynamic
            headers = ['framework', 'action', 'duration']
            for metric, stats in stats_per_action[0][2].items():
                headers.extend([f"{metric}_{stat}" for stat in stats.keys()])

            # Create a list of data rows
            writer.writerow(headers)

        for stat in stats_per_action:
            action, duration, metrics = stat
            row = [framework, action, duration]
            for metric, stats in metrics.items():
                row.extend([stats[stat] for stat in stats.keys()])
            writer.writerow(row)


def load_events():
    with open(EVENTS_FILE_PATH) as f:
        reader = csv.reader(f)
        next(reader, None)
        return list((line[0], int(float(line[1]))) for line in reader)


def main(framework, container):
    events = load_events()

    print(f'Pulling metrics for container: {container}')
    print(f'Events: {events}')

    start_time = events[0][1]
    end_time = events[len(events) - 1][1]

    print(f'Start pulling metrics: START: {start_time} - END {end_time}')
    # print(f'Metrics to query: {gauge_metric_names}')

    metric_values = query_metric_values(GAUGE_METRICS + COUNTER_METRICS, start_time, end_time, container)
    postprocess_and_upload(metric_values, events, framework)


if __name__ == "__main__":
    main(framework=sys.argv[1], container=sys.argv[2])