using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GateScript : MonoBehaviour
{
    
    private CapsuleCollider col;
    // Start is called before the first frame update
    void Start()
    {
        col = GetComponent<CapsuleCollider>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            SpaceShipController player = other.GetComponent<SpaceShipController>();
            player.SellMinerals();
        }
    }
}
