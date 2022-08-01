using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MineralScript : MonoBehaviour
{
    private SphereCollider col;
    public float amount;
    
    // Start is called before the first frame update
    void Start()
    {
        col = GetComponent<SphereCollider>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            SpaceManager spaceManager = GetComponentInParent<SpaceManager>();
            spaceManager.MineralTaken(gameObject);
        }
    }
}
