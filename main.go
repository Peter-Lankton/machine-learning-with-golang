package main

import (
	"encoding/csv"
	"fmt"
	"github.com/pkg/errors"
	"github.com/sajari/regression"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/palette"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf64"
	"image/color"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
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

// iqr finds the interquartile range
func iqr(a [][]float64, ql, qh float64, j int) (l, m, h float64) {
	var s []float64
	for _, row := range a {
		s = append(s, row[j])
	}
	sort.Float64s(s)
	idxLF64 := (ql / 100) * float64(len(s))
	idxHF64 := (qh / 100) * float64(len(s))

	// lower
	if math.Trunc(idxLF64) == idxLF64 {
		idxL := int(idxLF64)
		l = s[idxL-1]
	} else {
		idxL := int(idxLF64)
		l = (s[idxL] + s[idxL-1]) / float64(2)
	}

	// higher
	if math.Trunc(idxHF64) == idxHF64 {
		idxH := int(idxHF64)
		h = s[idxH-1]
	} else {
		idxH := int(idxHF64)
		h = (s[idxH] + s[idxH-1]) / float64(2)
	}

	// median
	if len(s)%2 == 0 {
		n := len(s)
		m = (s[n/2-1] + s[n/2+1]) / float64(2)
	} else {
		n := len(s)
		m = s[n/2]
	}

	return
}

func scale(a [][]float64, j int) {
	l, m, h := iqr(a, 0.25, 0.75, j)
	s := h - l
	if s == 0 {
		s = 1
	}

	for _, row := range a {
		row[j] = (row[j] - m) / s
	}
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

// mHandleErr is the error handler for the main function.
// If an error happens within the main function, it is not
// unexpected for a fatal error to be logged and for the program to immediately quit.
func mHandleErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// ingest is a function that ingests the file and outputs the header, data, and index.
func ingest(f io.Reader) (header []string, data [][]string, indices []map[string][]int, err error) {
	r := csv.NewReader(f)

	// handle header
	if header, err = r.Read(); err != nil {
		return
	}

	indices = make([]map[string][]int, len(header))
	var rowCount, colCount int = 0, len(header)
	for rec, err := r.Read(); err == nil; rec, err = r.Read() {
		if len(rec) != colCount {
			return nil, nil, nil, errors.Errorf("Expected Columns: %d. Got %d columns in row %d", colCount, len(rec), rowCount)
		}
		data = append(data, rec)
		for j, val := range rec {
			if indices[j] == nil {
				indices[j] = make(map[string][]int)
			}
			indices[j][val] = append(indices[j][val], rowCount)
		}
		rowCount++
	}
	return
}

// cardinality counts the number of unique values in a column.
// This assumes that the index i of indices represents a column.
func cardinality(indices []map[string][]int) []int {
	retVal := make([]int, len(indices))
	for i, m := range indices {
		retVal[i] = len(m)
	}
	return retVal
}

// mode finds the most common value for each variable
func mode(index []map[string][]int) []string {
	retVal := make([]string, len(index))
	for i, m := range index {
		var max int
		for k, v := range m {
			if len(v) > max {
				max = len(v)
				retVal[i] = k
			}
		}
	}
	return retVal
}

// imputeCategorical replaces "NA" with the mode of categorical values
func imputeCategorical(a string, col int, hdr []string, modes []string) string {
	if a != "NA" || a != "" {
		return a
	}
	switch hdr[col] {
	case "MSZoning", "BsmtFullBath", "BsmtHalfBath", "Utilities", "Functional", "Electrical", "KitchenQual", "SaleType", "Exterior1st", "Exterior2nd":
		return modes[col]
	}
	return a
}

// convert converts a string into a slice of floats
func convert(a string, isCat bool, index map[string][]int, varName string) ([]float64, []string) {
	if isCat {
		return convertCategorical(a, index, varName)
	}
	// here we deliberately ignore errors, because the zero value of float64 is well, zero.
	f, _ := strconv.ParseFloat(a, 64)
	return []float64{f}, []string{varName}
}

// convertCategorical is a basic function that encodes a categorical variable as a slice of floats.
// There are no smarts involved at the moment.
// The encoder takes the first value of the map as the default value, encoding it as a []float{0,0,0,...}
func convertCategorical(a string, index map[string][]int, varName string) ([]float64, []string) {
	retVal := make([]float64, len(index)-1)

	// important: Go actually randomizes access to maps, so we actually need to sort the keys
	// optimization point: this function can be made stateful.
	tmp := make([]string, 0, len(index))
	for k := range index {
		tmp = append(tmp, k)
	}

	// numerical "categories" should be sorted numerically
	tmp = tryNumCat(a, index, tmp)

	// find NAs and swap with 0
	var naIndex int
	for i, v := range tmp {
		if v == "NA" {
			naIndex = i
			break
		}
	}
	tmp[0], tmp[naIndex] = tmp[naIndex], tmp[0]

	// build the encoding
	for i, v := range tmp[1:] {
		if v == a {
			retVal[i] = 1
			break
		}
	}
	for i, v := range tmp {
		tmp[i] = fmt.Sprintf("%v_%v", varName, v)
	}

	return retVal, tmp[1:]
}

// hints is a slice of bools indicating whether it's a categorical variable
func clean(hdr []string, data [][]string, indices []map[string][]int, hints []bool, ignored []string) (int, int, []float64, []float64, []string, []bool) {
	modes := mode(indices)
	var Xs, Ys []float64
	var newHints []bool
	var newHdr []string
	var cols int

	for i, row := range data {

		for j, col := range row {
			if hdr[j] == "Id" { // skip id
				continue
			}
			if hdr[j] == "SalePrice" { // we'll put SalePrice into Ys
				cxx, _ := convert(col, false, nil, hdr[j])
				Ys = append(Ys, cxx...)
				continue
			}

			if inList(hdr[j], ignored) {
				continue
			}

			if hints[j] {
				col = imputeCategorical(col, j, hdr, modes)
			}
			cxx, newHdrs := convert(col, hints[j], indices[j], hdr[j])
			Xs = append(Xs, cxx...)

			if i == 0 {
				h := make([]bool, len(cxx))
				for k := range h {
					h[k] = hints[j]
				}
				newHints = append(newHints, h...)
				newHdr = append(newHdr, newHdrs...)
			}
		}
		// add bias

		if i == 0 {
			cols = len(Xs)
		}
	}
	rows := len(data)
	if len(Ys) == 0 { // it's possible that there are no Ys (i.e. the test.csv file)
		Ys = make([]float64, len(data))
	}
	return rows, cols, Xs, Ys, newHdr, newHints
}

// CEF is the conditional expectation function that finds the conditionally expected value when each of the values in the variable is held fixed.
func CEF(Ys []float64, col int, index []map[string][]int) map[string]float64 {
	retVal := make(map[string]float64)
	for k, v := range index[col] {
		var mean float64
		for _, i := range v {
			mean += Ys[i]
		}
		mean /= float64(len(v))
		retVal[k] = mean
	}
	return retVal
}

// skew returns the skewness of a column/variable
func skew(it [][]float64, col int) float64 {
	a := make([]float64, 0, len(it[0]))
	for _, row := range it {
		for _, col := range row {
			a = append(a, col)
		}
	}
	return stat.Skew(a, nil)
}

// log1pCol applies the log1p transformation on a column
func log1pCol(it [][]float64, col int) {
	for i := range it {
		it[i][col] = math.Log1p(it[i][col])
	}
}

// plotCEF plots the CEF. This is a simple plot with only the CEF.
// More advanced plots can be also drawn to expose more nuance in understanding the data.
func plotCEF(m map[string]float64) (*plot.Plot, error) {
	ordered := make([]string, 0, len(m))
	for k := range m {
		ordered = append(ordered, k)
	}
	sort.Strings(ordered)

	p := plot.New()
	points := make(plotter.XYs, len(ordered))
	for i, val := range ordered {
		// if val can be converted into a float, we'll use it
		// otherwise, we'll stick with using the index
		points[i].X = float64(i)
		if x, err := strconv.ParseFloat(val, 64); err == nil {
			points[i].X = x
		}

		points[i].Y = m[val]
	}
	if err := plotutil.AddLinePoints(p, "CEF", points); err != nil {
		return nil, err
	}
	return p, nil
}

// plotHist plots the histogram of a slice of float64s.
func plotHist(a []float64) (*plot.Plot, error) {
	h, err := plotter.NewHist(plotter.Values(a), 10)
	if err != nil {
		return nil, err
	}
	p := plot.New()

	h.Normalize(1)
	p.Add(h)
	return p, nil
}

type heatmap struct {
	x mat.Matrix
}

func (m heatmap) Dims() (c, r int)   { r, c = m.x.Dims(); return c, r }
func (m heatmap) Z(c, r int) float64 { return m.x.At(r, c) }
func (m heatmap) X(c int) float64    { return float64(c) }
func (m heatmap) Y(r int) float64    { return float64(r) }

type ticks []string

func (t ticks) Ticks(min, max float64) []plot.Tick {
	var retVal []plot.Tick
	for i := math.Trunc(min); i <= max; i++ {
		retVal = append(retVal, plot.Tick{Value: i, Label: t[int(i)]})
	}
	return retVal
}

func plotHeatMap(corr mat.Matrix, labels []string) (p *plot.Plot, err error) {
	pal := palette.Heat(48, 1)
	m := heatmap{corr}
	hm := plotter.NewHeatMap(m, pal)
	p = plot.New()
	hm.NaN = color.RGBA{0, 0, 0, 0} // black

	// add and adjust the prettiness of the chart
	p.Add(hm)
	p.X.Tick.Label.Rotation = 1.5
	p.Y.Tick.Label.Font.Size = 6
	p.X.Tick.Label.Font.Size = 6
	p.X.Tick.Label.XAlign = draw.XRight
	p.X.Tick.Marker = ticks(labels)
	p.Y.Tick.Marker = ticks(labels)

	// add legend
	l := plot.NewLegend()

	thumbs := plotter.PaletteThumbnailers(pal)
	for i := len(thumbs) - 1; i >= 0; i-- {
		t := thumbs[i]
		if i != 0 && i != len(thumbs)-1 {
			l.Add("", t)
			continue
		}
		var val float64
		switch i {
		case 0:
			val = hm.Min
		case len(thumbs) - 1:
			val = hm.Max
		}
		l.Add(fmt.Sprintf("%.2g", val), t)
	}

	// this is a hack. I place the legends between the axis and the actual heatmap
	// because if the legend is on the right, we'd need to create a custom canvas to take
	// into account the additional width of the legend.
	//
	// So instead, we shrink the legend width to fit snugly within the margins of the plot and the axes.
	l.Left = true
	l.XOffs = -5
	l.ThumbnailWidth = 5

	p.Legend = l
	return
}

func transform(it [][]float64, hdr []string, hints []bool) []int {
	var transformed []int
	for i, isCat := range hints {
		if isCat {
			continue
		}
		skewness := skew(it, i)
		if skewness > 0.75 {
			transformed = append(transformed, i)
			log1pCol(it, i)
		}
	}
	for i, h := range hints {
		if !h {
			scale(it, i)
		}
	}
	return transformed
}

func transform2(it [][]float64, hints []bool, previouslyTransformed []int) {
	for _, i := range previouslyTransformed {
		log1pCol(it, i)
	}
	for i, h := range hints {
		if !h {
			scale(it, i)
		}
	}
}

func runRegression(Xs [][]float64, Ys []float64, hdr []string) (r *regression.Regression, stdErr []float64) {
	r = new(regression.Regression)
	dp := make(regression.DataPoints, 0, len(Xs))
	for i, h := range hdr {
		r.SetVar(i, h)
	}
	for i, row := range Xs {
		dp = append(dp, regression.DataPoint(Ys[i], row))
	}
	r.Train(dp...)
	r.Run()

	// calculate StdErr
	var sseY float64
	sseX := make([]float64, len(hdr)+1)
	meanX := make([]float64, len(hdr)+1)
	for i, row := range Xs {
		pred, _ := r.Predict(row)
		sseY += (Ys[i] - pred) * (Ys[i] - pred)
		for j, c := range row {
			meanX[j+1] += c
		}
	}
	sseY /= float64(len(Xs) - len(hdr) - 1) // n - df ; df = len(hdr) + 1
	vecf64.ScaleInv(meanX, float64(len(Xs)))
	sseX[0] = 1
	for _, row := range Xs {
		for j, c := range row {
			sseX[j+1] += (c - meanX[j+1]) * (c - meanX[j+1])
		}
	}
	sseY = math.Sqrt(sseY)
	vecf64.Sqrt(sseX)
	vecf64.ScaleInvR(sseX, sseY)

	return r, sseX
}

func exploration() {
	f, err := os.Open("TikTok-toul_codes.csv")
	mHandleErr(err)
	hdr, data, indices, err := ingest(f)
	mHandleErr(err)

	fmt.Printf("Original Data: \nRows: %d, Cols: %d\n========\n", len(data), len(hdr))
	c := cardinality(indices)
	for i, h := range hdr {
		if datahints[i] {
			fmt.Printf("%v: %v\n", h, c[i])
		}
		if i >= 5 {
			fmt.Printf("    ⋮\n")
			break
		}
	}
	fmt.Println("")

	fmt.Printf("Building into matrices\n=============\n")
	rows, cols, XsBack, YsBack, newHdr, _ := clean(hdr, data, indices, datahints, nil)
	Xs := tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(XsBack))
	fmt.Printf("Xs %v: \n%1.1s\n", Xs.Shape(), Xs)
	fmt.Println("")

	ofInterest := 19 // variable of interest is in column 19
	cef := CEF(YsBack, ofInterest, indices)
	plt, err := plotCEF(cef)
	mHandleErr(err)
	plt.Title.Text = fmt.Sprintf("CEF for %v", hdr[ofInterest])
	plt.X.Label.Text = hdr[ofInterest]
	plt.Y.Label.Text = "Conditionally Expected House Price"
	mHandleErr(plt.Save(25*vg.Centimeter, 25*vg.Centimeter, "CEF.png"))

	hist, err := plotHist(YsBack)
	mHandleErr(err)
	hist.Title.Text = "Histogram of House Prices"
	mHandleErr(hist.Save(25*vg.Centimeter, 25*vg.Centimeter, "hist.png"))

	for i := range YsBack {
		YsBack[i] = math.Log1p(YsBack[i])
	}
	hist2, err := plotHist(YsBack)
	mHandleErr(err)
	hist2.Title.Text = "Histogram of House Prices (Processed)"
	mHandleErr(hist2.Save(25*vg.Centimeter, 25*vg.Centimeter, "hist2.png"))

	// figure out the correlation of things
	m64, err := tensor.ToMat64(Xs, tensor.UseUnsafe())
	mHandleErr(err)
	fmt.Printf("header : ", newHdr)
	fmt.Printf("m64: ", m64)

	//corr := stat.CorrelationMatrix(nil, m64, nil)
	//hm, err := plotHeatMap(corr, newHdr)
	//mHandleErr(err)
	//hm.Save(60*vg.Centimeter, 60*vg.Centimeter, "heatmap.png")
	//
	//// heatmaps are nice to look at, but are quite ridiculous.
	//var tba []struct {
	//	h1, h2 string
	//	corr   float64
	//}
	//for i, h1 := range newHdr {
	//	for j, h2 := range newHdr {
	//		if c := corr.At(i, j); math.Abs(c) >= 0.5 && h1 != h2 {
	//			tba = append(tba, struct {
	//				h1, h2 string
	//				corr   float64
	//			}{h1: h1, h2: h2, corr: c})
	//		}
	//	}
	//}
	//
	//fmt.Println("High Correlations:")
	//for _, a := range tba {
	//	fmt.Printf("\t%v-%v: %v\n", a.h1, a.h2, a.corr)
	//}
}

func main() {
	exploration()
}
