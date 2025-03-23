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
    '4. Output your recommendation as a single JSON object with exactly two keys:\n   - "exercise_recommendation": a JSON object with keys "recommendation_1", "recommendation_2", and "recommendation_3".\n   - "recommended_iot_devices": a JSON object mapping each device to its trigger.\n\nRemember to analyze the "Last 14" fields to determine if the current routine is effective or if a change is needed. If the user consistently provides low ratings back to back for the recommended exercises, avoid suggesting those again by either choosing a different option from the baseline list or by proposing an entirely new exercise idea that might better suit the users preferences. Also, consider the times at which the user exercises to see if there is a preferred time that should influence the recommendation.',
};

// Function to build the dynamic prompt from user input and combine it with system and fine-tuning parts
async function run(userInput) {
  console.log("User Input:", userInput);
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
  // Log the last few parts (e.g., last 3 parts)
  console.log("Last few prompt parts:", parts.slice(-3));
  const result = await model.generateContent({
    contents: [{ role: "user", parts }],
    generationConfig,
  });
  console.log("Result:", result.response.text());
  return result.response.text();
}

const app = express();
const port = process.env.PORT || 3001;

app.use(
  cors({
    origin: "*", // Relaxed for native apps
    methods: ["GET", "POST", "PUT", "DELETE"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);
app.use(express.json());

app.get("/", (req, res) => {
  res.send("Server is running!");
});

app.post("/recommendation", async (req, res) => {
  console.log("Received request:", req.body);
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

// New endpoint: Get Opinion on Last 14 Days Activity with Goals (Fine-Tuned)
app.post("/activityOpinion", async (req, res) => {
  try {
    console.log("Received request:", req.body);
    const userInput = req.body;
    const { last14Activities, last14Ratings, last14Times, goals } = userInput;

    // Define a fine-tuned system prompt that includes the user goals and specific tone/length instructions
    const systemPromptOpinion = {
      text:
        'You are "healthleh", a personal activity coach with a positive tone. Analyze the user\'s last 14 days activity data provided in the fields "last14Activities", "last14Ratings", and "last14Times", and consider the user goals provided in "goals". ' +
        "Provide a concise overall opinion on the user's current activity performance, with a strong focus on how well their activities align with their stated goals. " +
        "Identify any gaps between their current routine and their goals, and provide clear, actionable suggestions that directly help the user work towards achieving those goals. " +
        "Make sure your recommendations are specific and targeted, ensuring every suggestion is a step toward closing the gap between current performance and desired outcomes. " +
        'Output your recommendation as a single JSON object with exactly two keys: "activity_opinion" (a string summarizing your opinion) and "improvement_suggestions" (an array of strings, each being a suggestion). Limit the Activity opinion to 200 words and up to 5 suggestion 50 words each. ' +
        "Encourage any repeated exercises in succession which could be a sign of a routine and habit forming, and suggest ways to improve or change the routine to keep it fresh and engaging. " +
        "Remember to keep a positive and encouraging tone throughout, focusing on the progress made and the potential for further improvement.",
    };
    

    // Build dynamic prompt parts using available fields
    const dynamicParts = [];
    if (last14Activities) {
      dynamicParts.push({ text: `Last 14 Activities: ${last14Activities}` });
    }
    if (last14Ratings) {
      dynamicParts.push({ text: `Last 14 Ratings: ${last14Ratings}` });
    }
    if (last14Times) {
      dynamicParts.push({ text: `Last 14 Times: ${last14Times}` });
    }
    if (goals) {
      dynamicParts.push({ text: `User Goals: ${goals}` });
    }

    const parts = [systemPromptOpinion, ...dynamicParts];

    const result = await model.generateContent({
      contents: [{ role: "user", parts }],
      generationConfig: {
        temperature: 1,
        topP: 0.95,
        topK: 40,
        maxOutputTokens: 8192,
        responseMimeType: "application/json",
        responseSchema: {
          type: "object",
          properties: {
            activity_opinion: { type: "string" },
            improvement_suggestions: {
              type: "array",
              items: { type: "string" },
            },
          },
          required: ["activity_opinion", "improvement_suggestions"],
        },
      },
    });

    const opinionText = result.response.text();
    const opinion = JSON.parse(opinionText);
    res.json(opinion);
  } catch (error) {
    console.error("Error generating activity opinion:", error);
    res.status(500).json({ error: "Error generating activity opinion" });
  }
});

app.listen(port, () => {
  console.log(`Server is listening on port ${port}`);
});
