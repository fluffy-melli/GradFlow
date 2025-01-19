package GradFlow

import (
	"math"
	"strings"
)

func Vector(text string, vocabulary map[string]int) []float64 {
	words := strings.Fields(text)
	vector := make([]float64, len(vocabulary))
	for _, word := range words {
		if index, exists := vocabulary[word]; exists {
			vector[index]++
		}
	}
	return vector
}

func LogisticRegression(features []float64, weights []float64) float64 {
	var sum float64
	for i := 0; i < len(features); i++ {
		sum += features[i] * weights[i]
	}
	return 1 / (1 + math.Exp(-sum))
}

func GradientDescentOptimization(data [][]float64, labels []float64, weights []float64, learningRate float64, iterations int) []float64 {
	n := len(data)
	m := len(data[0])
	for i := 0; i < iterations; i++ {
		gradients := make([]float64, m)
		for j := 0; j < n; j++ {
			pred := LogisticRegression(data[j], weights)
			error := pred - labels[j]
			for k := 0; k < m; k++ {
				gradients[k] += error * data[j][k]
			}
		}
		for k := 0; k < m; k++ {
			weights[k] -= (learningRate / float64(n)) * gradients[k]
		}
	}
	return weights
}

type GradientDescent struct {
	Weights    []float64
	Vocabulary map[string]int
	TrainText  map[string]float64
}

func NewGradientDescent(text map[string]float64, word map[string]int, rate float64, epoch int) *GradientDescent {
	var m GradientDescent
	m.TrainText = text
	m.Vocabulary = word
	var TextLabel []float64
	var TextVector [][]float64
	for s, l := range m.TrainText {
		TextLabel = append(TextLabel, l)
		TextVector = append(TextVector, Vector(s, word))
	}
	m.Weights = GradientDescentOptimization(TextVector, TextLabel, make([]float64, len(word)), rate, epoch)
	return &m
}

func (m *GradientDescent) Predict(text string) float64 {
	vector := Vector(text, m.Vocabulary)
	return LogisticRegression(vector, m.Weights)
}
