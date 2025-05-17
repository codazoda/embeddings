package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

const openAIEndpoint = "https://api.openai.com/v1/embeddings"
const modelID = "text-embedding-3-small"

// stripEmbedding removes any line that starts with "<!-- embedding:"
func stripEmbedding(r io.Reader) (string, error) {
	var b bytes.Buffer
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(strings.TrimSpace(line), "<!-- embedding:") {
			continue
		}
		b.WriteString(line + "\n")
	}
	return b.String(), sc.Err()
}

type embReq struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type embResp struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

func getEmbedding(text string) ([]float64, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY not set")
	}
	payload, _ := json.Marshal(embReq{Input: text, Model: modelID})

	req, _ := http.NewRequest("POST", openAIEndpoint, bytes.NewReader(payload))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var res embResp
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return nil, err
	}
	if len(res.Data) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}
	return res.Data[0].Embedding, nil
}

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: mdembed <file.md>")
		os.Exit(1)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		panic(err)
	}
	defer f.Close()

	cleanText, err := stripEmbedding(f)
	if err != nil {
		panic(err)
	}

	vec, err := getEmbedding(cleanText)
	if err != nil {
		panic(err)
	}

	// print as JSON array
	out, _ := json.Marshal(vec)
	fmt.Println(string(out))
}
