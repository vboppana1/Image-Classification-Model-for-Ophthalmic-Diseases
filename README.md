<table width="100%">
  <tr>
    <td width="50%" align="center" style="padding-right: 5px;">
      <img src="https://github.com/user-attachments/assets/502217f0-f3f0-4144-9dee-87b6abb37c01" alt="Image 1" style="max-width: 100%;">
    </td>
    <td width="50%" align="center" style="padding-left: 5px;">
      <img src="https://github.com/user-attachments/assets/d691f79d-0e76-4205-b3dd-0accd8662220" alt="Image 2" style="max-width: 100%;">
    </td>
  </tr>
</table>

<p>The model was trained using a ResNet-18 architecture. Its optimal performance was achieved at <strong>Epoch 2</strong>, where it reached its lowest validation loss. The key performance metrics from this epoch are summarized below:</p>
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th align="center">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Lowest Validation Loss</strong></td>
      <td align="center">0.1832</td>
    </tr>
    <tr>
      <td><strong>Training Loss</strong></td>
      <td align="center">0.1896</td>
    </tr>
    <tr>
      <td><strong>Exact Match Accuracy</strong></td>
      <td align="center">55.43%</td>
    </tr>
  </tbody>
</table>

<p>The model's accuracy in predicting each individual disease at this optimal epoch is as follows:</p>
<table>
  <thead>
    <tr>
      <th>Disease</th>
      <th align="center">Validation Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Diabetes (D)</td>
      <td align="center">76.31%</td>
    </tr>
    <tr>
      <td>Glaucoma (G)</td>
      <td align="center">95.62%</td>
    </tr>
    <tr>
      <td>Cataract (C)</td>
      <td align="center">97.81%</td>
    </tr>
    <tr>
      <td>Age-Related Macular Degeneration (A)</td>
      <td align="center">96.25%</td>
    </tr>
     <tr>
      <td>Hypertension (H)</td>
      <td align="center">98.44%</td>
    </tr>
     <tr>
      <td>Pathological Myopia (M)</td>
      <td align="center">98.75%</td>
    </tr>
     <tr>
      <td>Other Diseases (O)</td>
      <td align="center">88.82%</td>
    </tr>
  </tbody>
</table>
