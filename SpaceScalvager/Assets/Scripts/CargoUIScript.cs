using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class CargoUIScript : MonoBehaviour
{
    public Slider slider;
    public TextMeshProUGUI text;

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
}
