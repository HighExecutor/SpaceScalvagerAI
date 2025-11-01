using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ShowMobileControl : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
#if UNITY_ANDROID || UNITY_IOS
        bool isMobile = true;
#else
        bool isMobile = Application.isMobilePlatform;
#endif
        gameObject.SetActive(isMobile);
        if (isMobile)
        {
            foreach (Transform child in transform)
            {
                child.gameObject.SetActive(true);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
