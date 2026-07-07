//go:build !darwin || !cgo

package infer

// infer_mlx_stub.go — MLX engine is Apple-Metal-only; on other platforms (or
// CGO_ENABLED=0 builds) these fail loudly and tests skip with a NOTICE.

import "fmt"

var errMLXUnsupported = fmt.Errorf("infer_mlx: MLX engine requires darwin+cgo (Apple Metal)")

func mlxAvailable() (bool, string) { return false, errMLXUnsupported.Error() }

func embedMLX(texts []string) ([][]float32, error) { return nil, errMLXUnsupported }

func judgeMLXBatch(a, b []string) ([][3]float64, error) { return nil, errMLXUnsupported }

func MLXVecsLoad(name string, flat []float32, n, dim int) error { return errMLXUnsupported }

func MLXVecsTopK(name string, q []float32, k int) ([]int32, []float32, error) {
	return nil, nil, errMLXUnsupported
}

func MLXVecsScore(name string, q []float32, n int) ([]float32, error) {
	return nil, errMLXUnsupported
}

func judgeMLX(oldSide, newSide string) ([3]float64, error) {
	return [3]float64{}, errMLXUnsupported
}
