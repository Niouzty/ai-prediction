package com.logistique.ml;

import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;

import java.util.LinkedHashMap;

public class ModelePluie {

    private final RowProcessor<Label> rowProcessor;

    public ModelePluie() {
        LabelFactory labelFactory = new LabelFactory();
        LinkedHashMap<String, FieldProcessor> fieldProcessors = new LinkedHashMap<>();

        fieldProcessors.put("jour_semaine", new IdentityProcessor("jour_semaine"));
        fieldProcessors.put("retard", new IdentityProcessor("retard"));

        FieldResponseProcessor<Label> responseProcessor =
                new FieldResponseProcessor<>("pluie", "non", labelFactory);

        this.rowProcessor = new RowProcessor<>(responseProcessor, fieldProcessors);
    }

    public RowProcessor<Label> obtenirRowProcessor() {
        return rowProcessor;
    }
}
