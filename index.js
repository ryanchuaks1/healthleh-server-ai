require("dotenv").config();
const cors = require("cors");
const express = require("express");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const fineTuningParts = require("./fine-tuning");

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("Error: GEMINI_API_KEY is not set in the environment.");
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);
const model = genAI.getGenerativeModel({
  model: "gemini-2.0-flash",
});

// config
const generationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 40,
  maxOutputTokens: 8192,
  responseMimeType: "application/json",
  responseSchema: {
    type: "object",
    properties: {
      exercise_recommendation: {
        type: "object",
        properties: {
          recommendation_1: { type: "string" },
          recommendation_2: { type: "string" },
          recommendation_3: { type: "string" },
        },
        required: ["recommendation_1", "recommendation_2", "recommendation_3"],
      },
      recommended_iot_devices: {
        type: "object",
        properties: {
          Phone: { type: "string" },
          Watch: { type: "string" },
        },
        required: ["Phone", "Watch"],
      },
    },
    required: ["exercise_recommendation", "recommended_iot_devices"],
  },
};

// Base system prompt
const systemPrompt = {
  text:
    'You are an exercise recommendation assistant. Based on the user\'s recent activity history provided in fields starting with "Last 14" (ordered from the furthest to most recent), perform the following tasks:\n\n' +
    '1. Analyze the user\'s history:\n   - "Last 14 Times": Time stamps of past exercises. Consider whether there is a time-of-day preference and how that might affect exercise performance.\n   - "Last 14 Distances": Distance data for each session.\n   - "Last 14 Exercises": The type of exercise performed (e.g., Longer Distance Walking, Climbing Stairs, etc.).\n   - "Last 14 Ratings": User satisfaction ratings (e.g., on a scale of 1-5).\n\n' +
    "2. Decide on one physical activity recommendation:\n   - Use these exercise suggestions as a baseline: Longer Distance Walking, Climbing Stairs, Jumping Jacks, Running, Burpees.\n   - Note: Exercises like Longer Distance Walking, Paced Walking, and Climbing Stairs are preferred when the user is farther from home, while Jumping Jacks, Running, and Burpees are better when closer.\n   - IMPORTANT: If the user's recent history shows consistent dissatisfaction (e.g., low ratings) with a particular exercise, do not recommend that exercise again. Instead, either select a different option from the baseline list or, if none seem acceptable, propose a new exercise idea that is reasonable given the context.\n\n" +
    "3. Choose an IoT trigger for reminders:\n   - Options include: Phone (full screen notification), Watch (vibrate and notify), Light (flash lights), Smart TV (display message \"It's time for your exercise\").\n   - To build habits, use the same reminder consistently unless the user's history indicates it's time for a change.\n\n" +
    '4. Output your recommendation as a single JSON object with exactly two keys:\n   - "exercise_recommendation": a JSON object with keys "recommendation_1", "recommendation_2", and "recommendation_3".\n   - "recommended_iot_devices": a JSON object mapping each device to its trigger.\n\nExample output:\n{\n  "exercise_recommendation": {\n    "recommendation_1": "Longer Distance Walking",\n    "recommendation_2": "Paced Walking",\n    "recommendation_3": "Climbing Stairs"\n  },\n  "recommended_iot_devices": {\n    "Phone": "full screen notification",\n    "Watch": "vibrate and notify"\n  }\n}\n\nRemember to analyze the "Last 14" fields to determine if the current routine is effective or if a change is needed. If the user consistently provides low ratings for the recommended exercises, avoid suggesting those again by either choosing a different option from the baseline list or by proposing an entirely new exercise idea that might better suit the users preferences. Also, consider the times at which the user exercises to see if there is a preferred time that should influence the recommendation.',
};

// Function to build the dynamic prompt from user input and combine it with system and fine-tuning parts
async function run(userInput) {
  const dynamicParts = [];
  if (userInput.timeTriggered) {
    dynamicParts.push({ text: `Time Triggered ${userInput.timeTriggered}` });
  }
  if (userInput.userDistanceFromHome) {
    dynamicParts.push({ text: `User Distance From Home ${userInput.userDistanceFromHome}` });
  }
  if (userInput.last14Distances) {
    dynamicParts.push({ text: `Last 14 Distance From Home ${userInput.last14Distances}` });
  }
  if (userInput.connectedIotDevices) {
    dynamicParts.push({ text: `Connected Iot Devices ${userInput.connectedIotDevices}` });
  }
  if (userInput.last14UsedIotDevices) {
    dynamicParts.push({ text: `Last 14 Used Iot Devices ${userInput.last14UsedIotDevices}` });
  }
  if (userInput.last14ActivityPerformed) {
    dynamicParts.push({ text: `Last 14 Activity Performed ${userInput.last14ActivityPerformed}` });
  }
  if (userInput.last14ActivityRating) {
    dynamicParts.push({ text: `Last 14 activity Rating ${userInput.last14ActivityRating}` });
  }

  const parts = [systemPrompt, ...fineTuningParts, ...dynamicParts];

  const result = await model.generateContent({
    contents: [{ role: "user", parts }],
    generationConfig,
  });
  return result.response.text();
}

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

app.use(express.json());
app.post("/recommendation", async (req, res) => {
  try {
    const userInput = req.body;
    const recommendationText = await run(userInput);
    const recommendation = JSON.parse(recommendationText);
    res.json(recommendation);
  } catch (error) {
    console.error("Error generating recommendation:", error);
    res.status(500).json({ error: "Error generating recommendation" });
  }
});

// New endpoint: Calculate Calories Burned
app.post("/calculateCalories", (req, res) => {
  try {
    const { exercise, duration, intensity } = req.body;
    if (!exercise || !duration || !intensity) {
      return res.status(400).json({ error: "Missing required parameters: exercise, duration, intensity" });
    }

    // Define base factors for common exercises (in kcal per minute)
    const baseFactors = {
      "Longer Distance Walking": 3.5,
      "Paced Walking": 3.0,
      "Climbing Stairs": 4.0,
      "Jumping Jacks": 8.0,
      Running: 9.0,
      Burpees: 10.0,
      Cycling: 7.5,
      Swimming: 8.5,
      Elliptical: 7.0,
      Rowing: 8.0,
      Yoga: 3.0,
      Pilates: 3.5,
      "Strength Training": 6.0,
      HIIT: 11.0,
      Dancing: 5.5,
      Hiking: 5.0,
    };
    
    const factor = baseFactors[exercise] || 5.0;
    function getIntensityMultiplier(intensityValue) {
      if (intensityValue <= 5) {
        return 0.6 + ((intensityValue - 1) * 0.4) / 4;
      } else {
        return 1.0 + ((intensityValue - 5) * 0.5) / 5;
      }
    }

    const multiplier = getIntensityMultiplier(Number(intensity));
    const caloriesBurned = Number(duration) * factor * multiplier;

    res.json({ caloriesBurned });
  } catch (error) {
    console.error("Error calculating calories:", error);
    res.status(500).json({ error: "Error calculating calories" });
  }
});

app.listen(port, () => {
  console.log(`Server is listening on port ${port}`);
});
