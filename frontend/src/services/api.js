import axios from "axios";
const API_URL = "https://aihealthassistant-ibpq.onrender.com";

export const predictDisease = async (symptoms) => {
  const response = await axios.post(
    `${API_URL}/predict`,
    {
      symptoms: symptoms,
    }
  );

  return response.data;
};