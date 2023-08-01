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
}
