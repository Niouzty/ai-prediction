package com.logistique.ml;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Préprocesseur pour convertir heure_depart en format numérique (heure_decimal)
 */
public class HeureDepartPreprocessor {

    /**
     * Convertit le fichier CSV en remplaçant heure_depart par heure_decimal
     * @param inputPath Chemin du fichier d'entrée
     * @param outputPath Chemin du fichier de sortie
     */
    public static void convertPreprocessor(Path inputPath, Path outputPath) {
        try {
            List<String> lines = Files.readAllLines(inputPath);
            List<String> convertedLines = new ArrayList<>();

            for (int i = 0; i < lines.size(); i++) {
                String line = lines.get(i);

                // Traiter le header
                if (i == 0) {
                    // Remplacer heure_depart par heure_decimal
                    line = line.replace("heure_depart", "heure_decimal");
                    convertedLines.add(line);
                } else {
                    // Traiter les données
                    String[] parts = line.split(",");
                    if (parts.length >= 2) {
                        // Convertir l'heure (ex: "08:00" -> 8.0, "09:45" -> 9.75)
                        String heureStr = parts[1];
                        double heureDecimal = convertHeureToDecimal(heureStr);

                        // Reconstruire la ligne avec heure_decimal à la place de heure_depart
                        StringBuilder newLine = new StringBuilder();
                        newLine.append(parts[0]); // id_course
                        newLine.append(",").append(heureDecimal); // heure_decimal
                        for (int j = 2; j < parts.length; j++) {
                            newLine.append(",").append(parts[j]);
                        }
                        convertedLines.add(newLine.toString());
                    }
                }
            }

            // Écrire le fichier converti
            Files.write(outputPath, convertedLines);
        } catch (IOException e) {
            throw new RuntimeException("Erreur lors du prétraitement du fichier", e);
        }
    }

    /**
     * Convertit une heure au format HH:MM en format décimal
     * Ex: "08:00" -> 8.0, "09:45" -> 9.75
     */
    private static double convertHeureToDecimal(String heure) {
        String[] parts = heure.split(":");
        int heures = Integer.parseInt(parts[0]);
        int minutes = Integer.parseInt(parts[1]);
        return heures + (minutes / 60.0);
    }
}
