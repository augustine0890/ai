# Go-Based PDF Compression Guide

Complete beginner-friendly guide for building and running the PDF compressor in Go on Windows and macOS.

## Table of Contents

1. [What is Go?](#what-is-go)
2. [Installation Guide](#installation-guide)
3. [Understanding the Code](#understanding-the-code)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Running the Compressor](#running-the-compressor)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

---

## What is Go?

**Go** (also called Golang) is a programming language created by Google. It's:
- **Simple**: Easy to learn syntax
- **Fast**: Compiles to native machine code
- **Efficient**: Great for system tools and command-line programs

Our PDF compressor is written in Go because it's fast and can easily call external tools like Poppler.

---

## Installation Guide

### Windows Setup

#### Step 1: Install Go

1. Visit [golang.org/dl](https://golang.org/dl)
2. Download **Windows Installer** (`.msi` file)
3. Run the installer and follow the prompts
4. Accept default installation path: `C:\Program Files\Go`
5. Restart your computer

**Verify installation:**
- Open Command Prompt (Win + R, type `cmd`)
- Type: `go version`
- You should see: `go version go1.21.x windows/amd64`

#### Step 2: Install Poppler

Poppler is a tool that converts PDF pages to images.

1. Open Command Prompt (Win + R, type `cmd`)
2. Run: `winget install oschwartz10612.Poppler`
3. Wait for installation to complete
4. Restart Command Prompt

**Verify installation:**
- Type: `pdftoppm -v`
- You should see version information

#### Step 3: Install a Text Editor (Optional but Recommended)

- **VS Code**: [code.visualstudio.com](https://code.visualstudio.com)
- **Notepad++**: [notepad-plus-plus.org](https://notepad-plus-plus.org)

### macOS Setup

#### Step 1: Install Homebrew (if not already installed)

Homebrew is a package manager for macOS.

1. Open Terminal (Cmd + Space, type `terminal`)
2. Copy and paste this command:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Press Enter and follow prompts

#### Step 2: Install Go

```bash
brew install go
```

**Verify installation:**
```bash
go version
```

#### Step 3: Install Poppler

```bash
brew install poppler
```

**Verify installation:**
```bash
pdftoppm -v
```

---

## Understanding the Code

### What Does the Compressor Do?

The compressor performs 3 main steps:

```
Input PDF (8.6 MB)
    ↓
[Step 1] Rasterize: Convert PDF pages to JPEG images at 65 DPI
    ↓
[Step 2] Recombine: Import images back into a PDF
    ↓
[Step 3] Optimize: Compress the PDF with pdfcpu
    ↓
Output PDF (637 KB)
```

### Code Breakdown

**File: `compress/main.go`**

```go
package main

import (
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "github.com/pdfcpu/pdfcpu/pkg/api"
    "github.com/pdfcpu/pdfcpu/pkg/pdfcpu/model"
)
```

**What this means:**
- `package main` - This is the main program
- `import` - Load external libraries (like importing tools)

```go
func main() {
    // Get file paths
    baseDir, _ := os.Getwd()
    inputFile := filepath.Join(baseDir, "..", "data", "wang.pdf")
    outputFile := filepath.Join(baseDir, "..", "data", "compressed_output.pdf")
```

**What this means:**
- `func main()` - The starting point of the program
- `os.Getwd()` - Get current directory
- `filepath.Join()` - Combine path parts (like building a file path)

```go
    // Get original file size
    inputInfo, err := os.Stat(inputFile)
    originalSize := inputInfo.Size()
    fmt.Printf("Original size: %d bytes\n", originalSize)
```

**What this means:**
- `os.Stat()` - Get file information
- `fmt.Printf()` - Print formatted text (like `print()` in Python)

```go
    // Rasterize pages using Poppler at 65 DPI
    cmd := exec.Command(
        "C:\\Users\\QuyNN8\\AppData\\...\\pdftoppm.exe",
        "-r", "65",           // DPI resolution
        "-jpeg",              // Output format
        "-jpegopt", "quality=60,progressive=y,optimize=y",
        inputFile,
        filepath.Join(tmpDir, "page"),
    )
    cmd.Run()
```

**What this means:**
- `exec.Command()` - Run an external program (pdftoppm)
- `-r 65` - Render at 65 DPI
- `-jpeg` - Save as JPEG format
- `quality=60` - JPEG compression quality (0-100)

```go
    // Convert images back to PDF
    api.ImportImagesFile(imagePaths, outputPDF, nil, conf)
```

**What this means:**
- `api.ImportImagesFile()` - Use pdfcpu library to create PDF from images
- `imagePaths` - List of image files to combine
- `outputPDF` - Where to save the result

### Algorithm Explanation

**Step 1: Rasterization (Converting PDF to Images)**

```
PDF Page (vector format)
    ↓
pdftoppm tool
    ↓
JPEG Image (65 DPI, quality 60)
```

- **DPI (Dots Per Inch)**: Controls resolution
  - 65 DPI = 637 KB output (lower quality, smaller file)
  - 100 DPI = 1.75 MB output (better quality, larger file)
  - 200 DPI = 4.7 MB output (high quality, large file)

- **JPEG Quality**: Controls compression
  - Quality 60 = More compression, more artifacts
  - Quality 75 = Balanced
  - Quality 90 = Less compression, better quality

**Step 2: Recombination (Images → PDF)**

```
JPEG Images (page-1.jpg, page-2.jpg, page-3.jpg)
    ↓
pdfcpu ImportImagesFile()
    ↓
PDF with embedded images
```

**Step 3: Optimization (Compress PDF)**

```
PDF with images
    ↓
pdfcpu Optimize()
    ↓
Compressed PDF (removes duplicates, optimizes streams)
```

---

## Step-by-Step Setup

### Windows

#### 1. Open Command Prompt

- Press `Win + R`
- Type `cmd`
- Press Enter

#### 2. Navigate to Project Directory

```bash
cd d:\source\ai\compress
```

**What this does:** Changes to the folder containing the Go code

#### 3. Download Dependencies

```bash
go mod tidy
```

**What this does:**
- Reads `go.mod` file
- Downloads required libraries from internet
- Creates `go.sum` file with checksums
- Takes 1-2 minutes first time

**Expected output:**
```
go: downloading github.com/pdfcpu/pdfcpu v0.11.1
go: downloading github.com/disintegration/imaging v1.6.2
...
```

#### 4. Build the Program

```bash
go build -o compress.exe
```

**What this does:**
- Compiles Go code into executable
- Creates `compress.exe` file
- Takes 10-30 seconds

**Expected output:**
```
(no output = success)
```

**Verify build succeeded:**
```bash
dir compress.exe
```

You should see the file listed.

### macOS

#### 1. Open Terminal

- Press `Cmd + Space`
- Type `terminal`
- Press Enter

#### 2. Navigate to Project Directory

```bash
cd /path/to/source/ai/compress
```

Replace `/path/to/source/ai` with your actual path.

#### 3. Download Dependencies

```bash
go mod tidy
```

#### 4. Build the Program

```bash
go build -o compress
```

**Verify build succeeded:**
```bash
ls -la compress
```

---

## Running the Compressor

### Windows

#### Method 1: From Command Prompt

```bash
cd d:\source\ai\compress
compress.exe
```

#### Method 2: Double-click

1. Open File Explorer
2. Navigate to `d:\source\ai\compress`
3. Double-click `compress.exe`
4. A Command Prompt window will open showing progress

### macOS

#### From Terminal

```bash
cd /path/to/source/ai/compress
./compress
```

### Expected Output

```
Original size: 8997544 bytes (8786.7 KB)
Rasterizing pages at 65 DPI...
Optimizing rasterized PDF...
New size: 652180 bytes (636.9 KB)
Reduction: 92.8%
```

### Check Output File

**Windows:**
```bash
dir d:\source\ai\data\compressed_output.pdf
```

**macOS:**
```bash
ls -lh /path/to/source/ai/data/compressed_output.pdf
```

---

## Configuration

### Adjusting DPI (Image Resolution)

**File:** `compress/main.go` (Line 32)

**Current:**
```go
"-r", "65",
```

**Change to:**
```go
"-r", "100",  // Higher DPI = better quality, larger file
```

**DPI Options:**
| DPI | Output Size | Quality | Use Case |
|-----|-------------|---------|----------|
| 65  | ~637 KB     | Low     | Email, web |
| 75  | ~820 KB     | Medium  | Balanced |
| 100 | ~1.75 MB    | Good    | Documents |
| 120 | ~2.3 MB     | High    | Printing |

### Adjusting JPEG Quality

**File:** `compress/main.go` (Line 34)

**Current:**
```go
"-jpegopt", "quality=60,progressive=y,optimize=y",
```

**Change to:**
```go
"-jpegopt", "quality=75,progressive=y,optimize=y",  // Higher = better quality
```

**Quality Options:**
| Quality | Output Size | Artifacts | Use Case |
|---------|-------------|-----------|----------|
| 60      | ~637 KB     | Visible   | Maximum compression |
| 70      | ~992 KB     | Minimal   | Balanced |
| 80      | ~1.2 MB     | None      | High quality |
| 90      | ~1.5 MB     | None      | Archive |

### Changing Poppler Path

**File:** `compress/main.go` (Line 30)

If Poppler is installed elsewhere, find the path:

**Windows:**
```bash
where pdftoppm
```

**macOS:**
```bash
which pdftoppm
```

Then update line 30 with the full path.

---

## Troubleshooting

### Problem: "go: command not found"

**Cause:** Go is not installed or not in PATH

**Solution:**
1. Install Go (see Installation Guide)
2. Restart Command Prompt/Terminal
3. Try again

### Problem: "pdftoppm: executable file not found"

**Cause:** Poppler is not installed or not in PATH

**Solution:**
1. Install Poppler (see Installation Guide)
2. Restart Command Prompt/Terminal
3. Verify: `pdftoppm -v`

### Problem: "missing go.sum entry"

**Cause:** Dependencies not downloaded

**Solution:**
```bash
go mod tidy
```

### Problem: Build fails with errors

**Solution:**
1. Make sure you're in `compress/` directory
2. Run `go mod tidy`
3. Try building again: `go build -o compress.exe`

### Problem: Output file is too large

**Solution:**
1. Reduce DPI (change 65 to 50)
2. Reduce JPEG quality (change 60 to 50)
3. Rebuild: `go build -o compress.exe`
4. Run again: `compress.exe`

### Problem: Output file is blurry

**Solution:**
1. Increase DPI (change 65 to 100)
2. Increase JPEG quality (change 60 to 75)
3. Rebuild and run

### Problem: "Error running pdftoppm"

**Cause:** Poppler path is wrong

**Solution:**
1. Find Poppler: `where pdftoppm` (Windows) or `which pdftoppm` (macOS)
2. Update line 30 in `main.go` with correct path
3. Rebuild: `go build -o compress.exe`

---

## Quick Reference

### Windows Commands

```bash
# Navigate to project
cd d:\source\ai\compress

# Download dependencies
go mod tidy

# Build
go build -o compress.exe

# Run
compress.exe

# Check output
dir d:\source\ai\data\compressed_output.pdf
```

### macOS Commands

```bash
# Navigate to project
cd /path/to/source/ai/compress

# Download dependencies
go mod tidy

# Build
go build -o compress

# Run
./compress

# Check output
ls -lh /path/to/source/ai/data/compressed_output.pdf
```

---

## Next Steps

1. **Try different DPI values** to find your quality/size balance
2. **Batch process** multiple PDFs by modifying `main.go`
3. **Explore pdfcpu** documentation for more features
4. **Learn Go** at [golang.org/doc](https://golang.org/doc)

---

## Additional Resources

- **Go Documentation:** [golang.org/doc](https://golang.org/doc)
- **pdfcpu Library:** [github.com/pdfcpu/pdfcpu](https://github.com/pdfcpu/pdfcpu)
- **Poppler Documentation:** [poppler.freedesktop.org](https://poppler.freedesktop.org)
- **Go Tutorial:** [golang.org/tour](https://golang.org/tour)
