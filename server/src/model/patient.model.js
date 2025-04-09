
import mongoose, { Schema } from "mongoose";

const patientSchema = new Schema({
    fullName: {
        type: String
    },
    mobile: {
        type: String
    },
    patientID: {
        type: String
    },
    hospitalID: {
        type: String
    }
}, { timestamps: true });

export const Patient = mongoose.model("Patient", patientSchema);