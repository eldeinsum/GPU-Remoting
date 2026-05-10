# GPU-Remoting Network Emulator

# Overview

The design of the network emulator aims to simulate network latency. Given that the bandwidth of shared memory significantly exceeds that of a typical network, we can effectively simulate network latency using shared memory without accounting for its inherent latency (I think this assumption is only valid when the amount of data transferred is large...). For a more detailed explanation of the implementation, please refer to Section 4.1 of our paper.

# Test

We use tests/cuda_api/test_stress.cu to test the correctness of the network emulator. The major cost of this test is network communication, which is the bottleneck of the system. We can use the network emulator to simulate network communication and test the correctness of the system.

## Experimental Conditions

- rtt: 100us
- bandwidth: 1Gbps

## Theoretical Calculation

The amount of data transferred in each round is approximately 16 MB. The transmission time is calculated as 125.1ms.

Our transmission was repeated 100 times, resulting in a total time difference of 12510 ms.

## Test Data

**without emulator** cost: 15975 ms

**latency=0** cost: 39282 ms

**with emulator** cost: 51741 ms