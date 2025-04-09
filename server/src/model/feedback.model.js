
import mongoose, { Schema } from "mongoose";

const feedbackSchema = new Schema({
    patientID: {
        type: String
    },
    hospitalID: {
        type: String
    },
    departmentId: {
        type: String
    },
    sentimentIndex: {
        type: Number
    },
    topic: {
        type: String
    },
    contentTypeIndex: {
        type: Number
    },
    textContent: {
        type: String
    },
    mediaContent: {
        type: String
    }
}, { timestamps: true });

export const Feedback = mongoose.model("Feedback", feedbackSchema);