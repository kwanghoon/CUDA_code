// common/variant.cuh
// 재사용 코어: 경량 Strategy — 커널 런처를 std::function으로 담아 이름으로 선택.
// 알고리즘마다 런치 시그니처 Sig가 다르다 (scan/reduction: void(const T*,T*,int)).
#pragma once

#include <functional>
#include <string>

template <typename Sig>
struct Variant {
    std::string          key;     // CLI 선택용 (-a)
    std::string          label;   // 표 출력용
    std::function<Sig>   launch;  // 함수포인터/람다 = 교체 가능한 전략
};
