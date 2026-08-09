import { useState, useEffect } from "react";
import { predictDisease } from "./services/api";
import symptomsList from "./data/symptoms";
import jsPDF from "jspdf";

import {
  Container,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  Autocomplete,
} from "@mui/material";

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedSymptoms, setSelectedSymptoms] = useState([]);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");

  // Load prediction history
  useEffect(() => {
    const savedHistory = localStorage.getItem("predictionHistory");

    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch (error) {
        console.error("Error loading history:", error);
      }
    }
  }, []);

  // Open Google Maps
  const openMaps = () => {
    if (prediction?.google_maps) {
      window.open(prediction.google_maps, "_blank");
    } else {
      window.open(
        "https://www.google.com/maps/search/hospitals+near+me",
        "_blank"
      );
    }
  };

  // Predict disease
  const handlePredict = async () => {
    setError("");

    if (selectedSymptoms.length === 0) {
      setError("Please select at least one symptom.");
      return;
    }

    if (selectedSymptoms.length > 5) {
      setError("Please select a maximum of 5 symptoms.");
      return;
    }

    try {
      setLoading(true);

      const result = await predictDisease(selectedSymptoms);

      console.log("Prediction result:", result);

      setPrediction(result);

      // Create history record
      const newRecord = {
        date: new Date().toLocaleString(),
        disease: result.predicted_disease,
        symptoms: selectedSymptoms.join(", "),
      };

      // Add newest prediction first
      const updatedHistory = [newRecord, ...history];

      setHistory(updatedHistory);

      // Save history in browser
      localStorage.setItem(
        "predictionHistory",
        JSON.stringify(updatedHistory)
      );
    } catch (error) {
      console.error("Prediction error:", error);

      setError(
        "Unable to connect to the backend. Please make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  // Reset prediction
  const resetPrediction = () => {
    setPrediction(null);
    setSelectedSymptoms([]);
    setError("");
  };

  // Download PDF report
  const downloadReport = () => {
    if (!prediction) {
      return;
    }

    const doc = new jsPDF();

    let y = 20;

    // Title
    doc.setFontSize(20);
    doc.text("AI Health Report", 20, y);

    y += 15;

    // Date
    doc.setFontSize(12);

    doc.text(
      `Date: ${new Date().toLocaleDateString()}`,
      20,
      y
    );

    y += 10;

    // Disease
    doc.text(
      `Disease: ${prediction.predicted_disease || "Not available"}`,
      20,
      y
    );

    y += 10;

    // Confidence
    if (prediction.confidence) {
      doc.text(
        `Confidence: ${prediction.confidence}%`,
        20,
        y
      );

      y += 10;
    }

    // Doctor
    const doctorName =
      typeof prediction.doctor === "object"
        ? prediction.doctor?.name
        : prediction.doctor || "General Physician";

    const doctorSpecialization =
      typeof prediction.doctor === "object"
        ? prediction.doctor?.specialization
        : prediction.specialization || "General Medicine";

    doc.text(`Doctor: ${doctorName}`, 20, y);

    y += 10;

    doc.text(
      `Specialization: ${doctorSpecialization}`,
      20,
      y
    );

    y += 15;

    // Description
    doc.text("Description:", 20, y);

    y += 8;

    const description =
      prediction.description || "Not available.";

    const descriptionLines = doc.splitTextToSize(
      description,
      170
    );

    doc.text(descriptionLines, 20, y);

    y += descriptionLines.length * 6 + 10;

    // Precautions
    doc.text("Precautions:", 20, y);

    const precautions = prediction.precautions || [];

    precautions.forEach((item) => {
      y += 8;

      if (y > 280) {
        doc.addPage();
        y = 20;
      }

      doc.text("- " + item, 25, y);
    });

    y += 12;

    // Diet
    doc.text("Diet:", 20, y);

    const diet = prediction.diet || [];

    diet.forEach((item) => {
      y += 8;

      if (y > 280) {
        doc.addPage();
        y = 20;
      }

      doc.text("- " + item, 25, y);
    });

    y += 12;

    // Workout
    doc.text("Workout:", 20, y);

    const workout = prediction.workout || [];

    workout.forEach((item) => {
      y += 8;

      if (y > 280) {
        doc.addPage();
        y = 20;
      }

      doc.text("- " + item, 25, y);
    });

    y += 12;

    // Medication
    doc.text("Medication:", 20, y);

    const medication = prediction.medication || [];

    medication.forEach((item) => {
      y += 8;

      if (y > 280) {
        doc.addPage();
        y = 20;
      }

      doc.text("- " + item, 25, y);
    });

    // Save PDF
    doc.save("AI_Health_Report.pdf");
  };

  return (
    <Container
      maxWidth="md"
      sx={{
        mt: { xs: 2, sm: 5 },
        mb: 5,
        px: { xs: 1.5, sm: 3 },
      }}
    >
      <Card
        sx={{
          borderRadius: 4,
          boxShadow: 6,
          p: 3,
          background:
            "linear-gradient(135deg, #ffffff, #f1f8ff)",
        }}
      >
        <CardContent>

          {/* Title */}
          <Typography
            variant="h3"
            align="center"
            gutterBottom
            sx={{
              fontWeight: "bold",
              color: "#1565c0",
              fontSize: {
                xs: "2rem",
                sm: "2.8rem",
                md: "3rem",
              },
            }}
          >
            🩺 AI Health Assistant
          </Typography>

          <Typography
            align="center"
            color="text.secondary"
          >
            Select your symptoms below
          </Typography>

          {/* Symptoms */}
          <Autocomplete
            multiple
            limitTags={3}
            options={symptomsList}
            value={selectedSymptoms}
            onChange={(event, newValue) => {
              setSelectedSymptoms(newValue);
            }}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Symptoms"
                placeholder="Choose symptoms"
                margin="normal"
              />
            )}
          />

          {/* Error */}
          {error && (
            <Typography
              color="error"
              sx={{
                mt: 2,
                textAlign: "center",
                fontWeight: "bold",
              }}
            >
              ⚠️ {error}
            </Typography>
          )}

          {/* Predict Button */}
          <Button
            fullWidth
            variant="contained"
            size="large"
            sx={{
              mt: 2,
              py: 1.5,
              fontSize: "1.1rem",
              fontWeight: "bold",
              borderRadius: 3,
            }}
            onClick={handlePredict}
            disabled={loading}
          >
            🔍 Predict Disease
          </Button>

          {/* Loading */}
          {loading && (
            <div
              style={{
                textAlign: "center",
                marginTop: 20,
              }}
            >
              <CircularProgress />
            </div>
          )}

          {/* Prediction Result */}
          {prediction && (
            <Card
              sx={{
                mt: 4,
                backgroundColor: "#f8f9fa",
                overflow: "hidden",
              }}
            >
              <CardContent>

                {/* Disease */}
                <Typography variant="h4">
                  🦠 {prediction.predicted_disease}
                </Typography>

                {/* Confidence */}
                {prediction.confidence && (
                  <Typography
                    color="primary"
                    variant="h6"
                    sx={{ mt: 1 }}
                  >
                    Confidence: {prediction.confidence}%
                  </Typography>
                )}

                {/* Doctor */}
                <Typography
                  variant="h5"
                  sx={{ mt: 3 }}
                >
                  👨‍⚕️ Recommended Doctor
                </Typography>

                <Typography variant="h6">
                  {typeof prediction.doctor === "object"
                    ? prediction.doctor?.name
                    : prediction.doctor ||
                      "General Physician"}
                </Typography>

                <Typography color="text.secondary">
                  {typeof prediction.doctor === "object"
                    ? prediction.doctor?.specialization
                    : prediction.specialization ||
                      "General Medicine"}
                </Typography>

                {/* Description */}
                <Typography sx={{ mt: 3 }}>
                  {prediction.description ||
                    "No description available."}
                </Typography>

                {/* Precautions */}
                <Typography
                  variant="h6"
                  sx={{ mt: 3 }}
                >
                  ⚠️ Precautions
                </Typography>

                <List>
                  {(prediction.precautions || []).map(
                    (item, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={item}
                        />
                      </ListItem>
                    )
                  )}
                </List>

                {/* Diet */}
                <Typography variant="h6">
                  🥗 Recommended Diet
                </Typography>

                <List>
                  {(prediction.diet || []).map(
                    (item, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={item}
                        />
                      </ListItem>
                    )
                  )}
                </List>

                {/* Workout */}
                <Typography variant="h6">
                  🏃 Workout
                </Typography>

                <List>
                  {(prediction.workout || []).map(
                    (item, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={item}
                        />
                      </ListItem>
                    )
                  )}
                </List>

                {/* Medication */}
                <Typography variant="h6">
                  💊 Medication
                </Typography>

                <List>
                  {(prediction.medication || []).map(
                    (item, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={item}
                        />
                      </ListItem>
                    )
                  )}
                </List>

                {/* PDF */}
                <Button
                  fullWidth
                  variant="contained"
                  color="success"
                  sx={{ mt: 3 }}
                  onClick={downloadReport}
                >
                  📄 Download PDF Report
                </Button>

                {/* Google Maps */}
                <Button
                  fullWidth
                  variant="outlined"
                  color="primary"
                  sx={{ mt: 2 }}
                  onClick={openMaps}
                >
                  🏥 Find Nearby Hospitals
                </Button>

                {/* New Prediction */}
                <Button
                  fullWidth
                  variant="outlined"
                  sx={{ mt: 2 }}
                  onClick={resetPrediction}
                >
                  🔄 New Prediction
                </Button>

              </CardContent>
            </Card>
          )}

          {/* Prediction History */}
          {history.length > 0 && (
            <Card sx={{ mt: 4 }}>
              <CardContent>

                <Typography
                  variant="h5"
                  gutterBottom
                >
                  📋 Prediction History
                </Typography>

                <List>
                  {history.map((record, index) => (
                    <ListItem
                      key={index}
                      divider
                    >
                      <ListItemText
                        primary={record.disease}
                        secondary={
                          <>
                            <div>
                              Date: {record.date}
                            </div>

                            <div>
                              Symptoms: {record.symptoms}
                            </div>
                          </>
                        }
                      />
                    </ListItem>
                  ))}
                </List>

              </CardContent>
            </Card>
          )}

        </CardContent>
      </Card>
    </Container>
  );
}

export default App;
