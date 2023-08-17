using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class CargoUIScript : MonoBehaviour
{
    public Slider slider;
    public TextMeshProUGUI text;

    public TextMeshProUGUI credits;
    public TextMeshProUGUI modelControl;
    public TextMeshProUGUI controlsHelp;
    public TextMeshProUGUI timestampBar;
    public TextMeshProUGUI statsNamesBar;
    public TextMeshProUGUI statsResultBar;


    public void SetMaxCargo(int maxCargo)
    {
        slider.maxValue = maxCargo;
        slider.value = Mathf.Min(maxCargo, slider.value);
    }

    public void SetCargo(float curCargo)
    {
        slider.value = curCargo;
        SetText(curCargo);
    }

    public void SetText(float curCargo)
    {
        text.text = (int) curCargo + " / " + slider.maxValue;
    }

    public void SetCredutValue(float curCredits)
    {
        credits.text = (int)curCredits + " Cr";
    }

    public void SetModelControl(int mode)
    {
        if (mode == 0)
        {
            modelControl.text = "Default";
        }
        if (mode == 1)
        {
            modelControl.text = "Player";
        }
        if (mode == 2)
        {
            modelControl.text = "AI";
        }
    }

    public void SetHelpEnabled(bool openHelp)
    {
        controlsHelp.gameObject.SetActive(openHelp);
    }

    public void SetTimestepsBar(int curStep, int maxStep)
    {
        string result = "" + curStep;
        if (maxStep > 0)
        {
            result += " / " + maxStep;
        }
        timestampBar.SetText(result);
    }

    public void SetShipsNames(string[] shipsNames)
    {
        string ids = "";
        for (int i = 0; i < shipsNames.Length; i++)
        {
            ids += shipsNames[i];
            if (i < shipsNames.Length - 1)
            {
                ids += "\n";
            }
        }
        statsNamesBar.SetText(ids);
    }

    public void SetStats(int[] cargos, int[] credits)
    {
        string values = "";
        for (int i = 0; i < cargos.Length; i++)
        {
            values += credits[i] + "|" + cargos[i];
            if (i < cargos.Length -1)
            {
                values += "\n";
            }
        }
        statsResultBar.SetText(values);
    }
}
