// common/cli.cuh
// 재사용 코어: 플래그식 CLI 파싱 (-n / -i / -a / -l / -h).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct BenchOptions {
    long n = (1L << 24);               // -n: 문제 크기
    int  iters = 100;                  // -i: 측정 반복
    std::vector<std::string> algoKeys; // -a: 실행할 변형 key (비면 전체)
    bool listOnly = false;             // -l
    bool showHelp = false;             // -h
    bool valid = true;
};

class CliParser {
public:
    static void printUsage(const char* prog) {
        std::printf("Usage: %s [-n N] [-i ITERS] [-a KEY]... [-l] [-h]\n", prog);
        std::printf("  -n N       문제 크기 (기본 1<<24)\n");
        std::printf("  -i ITERS   측정 반복 (기본 100)\n");
        std::printf("  -a KEY     실행할 변형 key (반복 지정 가능; 미지정 시 전체)\n");
        std::printf("  -l         변형 목록 출력 후 종료\n");
        std::printf("  -h         도움말\n");
    }

    static BenchOptions parse(int argc, char** argv) {
        BenchOptions o;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "-n") o.n     = std::atol(value(argc, argv, i, o));
            else if (a == "-i") o.iters = std::atoi(value(argc, argv, i, o));
            else if (a == "-a") o.algoKeys.push_back(value(argc, argv, i, o));
            else if (a == "-l") o.listOnly = true;
            else if (a == "-h" || a == "--help") o.showHelp = true;
            else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); o.valid = false; }
            if (!o.valid) break;
        }
        return o;
    }

private:
    static const char* value(int argc, char** argv, int& i, BenchOptions& o) {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); o.valid = false; return "0"; }
        return argv[++i];
    }
};
