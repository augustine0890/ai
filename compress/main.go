package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/pdfcpu/pdfcpu/pkg/api"
	"github.com/pdfcpu/pdfcpu/pkg/pdfcpu/model"
)

func main() {
	// Get absolute paths
	baseDir, _ := os.Getwd()
	inputFile := filepath.Join(baseDir, "..", "data", "wang.pdf")
	outputFile := filepath.Join(baseDir, "..", "data", "compressed_output.pdf")
	tmpDir := filepath.Join(baseDir, "tmp_pages")

	_ = os.RemoveAll(tmpDir)
	_ = os.MkdirAll(tmpDir, 0755)

	// Get original file size
	inputInfo, err := os.Stat(inputFile)
	if err != nil {
		fmt.Printf("Error reading input file: %v\n", err)
		os.Exit(1)
	}
	originalSize := inputInfo.Size()
	fmt.Printf("Original size: %d bytes (%.1f KB)\n", originalSize, float64(originalSize)/1024)

	// Rasterize pages using Poppler (pdftoppm) at 65 DPI
	fmt.Println("Rasterizing pages at 65 DPI...")
	cmd := exec.Command(
		"C:\\Users\\QuyNN8\\AppData\\Local\\Microsoft\\WinGet\\Packages\\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\\poppler-25.07.0\\Library\\bin\\pdftoppm.exe",
		"-r", "65",
		"-jpeg",
		"-jpegopt", "quality=60,progressive=y,optimize=y",
		inputFile,
		filepath.Join(tmpDir, "page"),
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Printf("Error running pdftoppm: %v\n", err)
		os.Exit(1)
	}

	// Convert images back to PDF
	images, _ := filepath.Glob(filepath.Join(tmpDir, "page-*.jpg"))
	if len(images) == 0 {
		fmt.Println("No rasterized pages found")
		os.Exit(1)
	}

	intermediatePDF := filepath.Join(tmpDir, "rasterized.pdf")
	if err := imagesToPDF(images, intermediatePDF); err != nil {
		fmt.Printf("Error creating rasterized PDF: %v\n", err)
		os.Exit(1)
	}

	// Optimize rasterized PDF
	fmt.Println("Optimizing rasterized PDF...")
	conf := model.NewDefaultConfiguration()
	conf.ValidationMode = model.ValidationRelaxed
	if err := api.OptimizeFile(intermediatePDF, outputFile, conf); err != nil {
		fmt.Printf("Error optimizing PDF: %v\n", err)
		os.Exit(1)
	}

	// Check result
	outputInfo, err := os.Stat(outputFile)
	if err != nil {
		fmt.Printf("Error reading output file: %v\n", err)
		os.Exit(1)
	}
	newSize := outputInfo.Size()
	reduction := (1 - float64(newSize)/float64(originalSize)) * 100

	fmt.Printf("New size: %d bytes (%.1f KB)\n", newSize, float64(newSize)/1024)
	fmt.Printf("Reduction: %.1f%%\n", reduction)
}

func imagesToPDF(imagePaths []string, outputPDF string) error {
	conf := model.NewDefaultConfiguration()
	conf.ValidationMode = model.ValidationRelaxed
	return api.ImportImagesFile(imagePaths, outputPDF, nil, conf)
}
