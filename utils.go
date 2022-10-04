package main

import (
	"math/rand"
	"sort"
	"strconv"
)

func tryNumCat(a string, index map[string][]int, catStrs []string) []string {
	isNumCat := true
	cats := make([]int, 0, len(index))
	for k := range index {
		i64, err := strconv.ParseInt(k, 10, 64)
		if err != nil && k != "NA" {
			isNumCat = false
			break
		}
		cats = append(cats, int(i64))
	}

	if isNumCat {
		sort.Ints(cats)
		for i := range cats {
			catStrs[i] = strconv.Itoa(cats[i])
		}
		if _, ok := index["NA"]; ok {
			catStrs[0] = "NA" // there are no negative numerical categories
		}
	} else {
		sort.Strings(catStrs)
	}
	return catStrs
}

func inList(a string, l []string) bool {
	for _, v := range l {
		if a == v {
			return true
		}
	}
	return false
}

func scaleStd(a [][]float64, j int) {
	var mean, variance, n float64
	for _, row := range a {
		mean += row[j]
		n++
	}
	mean /= n
	for _, row := range a {
		variance += (row[j] - mean) * (row[j] - mean)
	}
	variance /= (n - 1)

	for _, row := range a {
		row[j] = (row[j] - mean) / variance
	}
}

func shuffle(a [][]float64, b []float64) {
	for i := len(a) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
		b[i], b[j] = b[j], b[i]
	}
}
