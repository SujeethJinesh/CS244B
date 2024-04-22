from collections import defaultdict

def parse_logs(log_file):
    test_cases = {}
    current_test_case = None

    with open(log_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Test Case:"):
                current_test_case = line.split(":")[1].strip()
                test_cases[current_test_case] = defaultdict(list)
            elif line.startswith("NODE"):
                parts = line.split()
                node_num = int(parts[1])
                test_cases[current_test_case][node_num].append(line)

    return test_cases

def check_logs(logs, ref_logs):
    for test_case, node_logs in ref_logs.items():
        test_case_passed = True
        if test_case not in logs:
            print(f"Test case {test_case} not found in log file.")
            continue
        
        for node, ref in node_logs.items():
            log = logs[test_case][node]
            j = 0
            for i, ref_line in enumerate(ref):
                if i >= len(log):
                    print(f"Missing log for test case {test_case}: {ref_line}")
                    test_case_passed = False
                    break
                while j < len(log):
                    if ref_line == log[j]:
                        break
                    j += 1
                if j >= len(log):
                    print(f"Mismatch in test case {test_case}: Expected '{ref_line}', it's missing")
                    test_case_passed = False
        if test_case_passed:
            print(f"Test case {test_case} passed")


if __name__ == "__main__":
    log_file = "log.txt"
    ref_file = "ref.txt"

    logs = parse_logs(log_file)
    ref_logs = parse_logs(ref_file)

    check_logs(logs, ref_logs)